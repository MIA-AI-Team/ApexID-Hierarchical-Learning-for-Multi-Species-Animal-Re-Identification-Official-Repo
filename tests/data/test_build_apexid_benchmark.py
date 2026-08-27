import copy
import csv
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "data" / "build_apexid_benchmark.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_apexid_benchmark", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(path, identity, dataset):
    return {"path": path, "identity": identity, "dataset": dataset}


def _synthetic_input():
    rows = []
    for identity in ("afia", "kera"):
        for session in (1, 2, 3):
            rows.append(
                _row(
                    f"images/bristol/{identity}-{session}-img-1.jpg",
                    identity,
                    "bristol_gorillas",
                )
            )
    for identity, camera in (("AP03", "R066"), ("DU40", "R030")):
        for date in ("20220101", "20220102", "20220103"):
            rows.append(
                _row(
                    f"images/spac/{identity}/{camera}_{date}_001_0.png",
                    identity,
                    "spac_wild",
                )
            )
    for identity in ("ELPephants_1002", "ELPephants_1009"):
        for date in ("7Mar2015", "8Sep2016", "Jan2003"):
            rows.append(
                _row(
                    f"images/elephants/{identity}_{date}.png",
                    identity,
                    "elephants",
                )
            )
    for index in range(3):
        rows.append(_row(f"images/gcn/1/IMG_{index}.jpg", "1", "gcn_id"))
    return rows


def _build(tmp_path, seed=42):
    mod = _load_module()
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "merged.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=mod.INPUT_FIELDS)
        writer.writeheader()
        writer.writerows(_synthetic_input())
    config = mod.BenchmarkConfig(
        input_path=source,
        output_dir=tmp_path / "output",
        seed=seed,
        num_stream_orders=2,
        dataset_to_species=mod.DEFAULT_DATASET_TO_SPECIES,
    )
    rows, warnings = mod.load_and_validate_csv(source, config.dataset_to_species)
    benchmark = mod.build_protocol(rows, mod.ProtocolConfig("open40", 0.40), config)
    benchmark["warnings"] = warnings + benchmark["warnings"]
    return mod, config, rows, benchmark


def test_encounter_parsers_cover_documented_and_conservative_forms():
    mod = _load_module()
    assert mod.infer_spac_encounter("AP03", "R066_20221118_171_2151.png") == (
        "spac_wild:AP03:R066_20221118",
        True,
    )
    assert mod.infer_spac_encounter("OE00", "Tm002_20220227_002_1002.png") == (
        "spac_wild:OE00:Tm002_20220227",
        True,
    )
    assert mod.infer_bristol_encounter("afia", "afia-5-img-78.jpg") == (
        "bristol_gorillas:afia:session_5",
        True,
    )
    assert mod.infer_bristol_encounter("touni", "touni-ggl-trial-img-95.jpg") == (
        "bristol_gorillas:touni:trial",
        True,
    )
    assert mod.infer_elephant_encounter("ELPephants_1909", "1909_Jamila_13Oct2016.png") == (
        "elephants:ELPephants_1909:2016-10-13",
        True,
    )
    assert mod.infer_elephant_encounter("ELPephants_35", "35_Malak_left_Jan2003.png") == (
        "elephants:ELPephants_35:2003-01",
        True,
    )
    assert mod.infer_elephant_encounter("ELPephants_492", "492_Barrymore.png") == (
        "elephants:ELPephants_492:unknown_date",
        False,
    )


def test_two_and_three_encounter_known_splits_are_encounter_disjoint():
    mod = _load_module()
    rows = [
        {"path": f"p{i}", "encounter_id": f"enc{i}", "dataset": "x", "identity": "id"}
        for i in range(3)
    ]
    two = mod.split_known_identity_encounters(rows[:2], 0.60, 42, ("x", "id"))
    assert len(two["database"]) == 1
    assert not two["development"]
    assert len(two["final_test"]) == 1
    three = mod.split_known_identity_encounters(rows, 0.60, 42, ("x", "id"))
    assert all(three[partition] for partition in ("database", "development", "final_test"))
    encounters = [
        row["encounter_id"]
        for partition_rows in three.values()
        for row in partition_rows
    ]
    assert sorted(encounters) == ["enc0", "enc1", "enc2"]


def test_protocol_isolation_public_schema_and_streams(tmp_path):
    _mod, _config, rows, benchmark = _build(tmp_path)
    assert benchmark["validation_report"]["status"] == "PASS"
    assert {row["path"] for row in benchmark["final_test"] if row["dataset"] == "gcn_id"} == {
        row["path"] for row in rows if row["dataset"] == "gcn_id"
    }
    assert not any(row["dataset"] == "gcn_id" for row in benchmark["database"])
    assert not any(row["dataset"] == "gcn_id" for row in benchmark["development"])
    forbidden = {"identity", "original_identity", "true_identity", "scenario", "is_known"}
    assert not forbidden & set(benchmark["final_test"][0])
    final_ids = {row["image_id"] for row in benchmark["final_test"]}
    for stream in benchmark["streams"].values():
        assert {row["image_id"] for row in stream} == final_ids
        assert [int(row["arrival_index"]) for row in stream] == list(range(1, len(stream) + 1))


def test_determinism_and_validation_detects_path_leakage(tmp_path):
    mod, config, rows, first = _build(tmp_path / "one")
    _mod, _config, _rows, second = _build(tmp_path / "two")
    assert first["final_test"] == second["final_test"]
    assert first["split_manifest"] == second["split_manifest"]
    assert first["streams"] == second["streams"]

    leaking = copy.deepcopy(first)
    leaking["development"].append(dict(leaking["final_test"][0]))
    with pytest.raises(ValueError, match="Path leakage"):
        mod.validate_benchmark(leaking, rows, config)


def test_write_outputs_keeps_public_and_private_contracts(tmp_path):
    mod, config, rows, benchmark = _build(tmp_path)
    benchmark["validation_report"] = mod.validate_benchmark(benchmark, rows, config)
    destination = mod.write_protocol(benchmark, config, "abc123")
    expected = {
        "database.csv",
        "development.csv",
        "development_ground_truth.csv",
        "final_test.csv",
        "final_test_no_species.csv",
        "final_test_ground_truth.csv",
        "known_identity_registry.csv",
        "private_identity_mapping.csv",
        "split_manifest.csv",
        "prediction_template.csv",
        "split_summary.csv",
        "split_summary.json",
        "config.json",
        "validation_report.json",
        "streams",
    }
    assert expected <= {path.name for path in destination.iterdir()}
    with (destination / "final_test.csv").open(newline="", encoding="utf-8") as handle:
        assert tuple(csv.DictReader(handle).fieldnames) == mod.PUBLIC_FINAL_FIELDS
    with (destination / "final_test_ground_truth.csv").open(newline="", encoding="utf-8") as handle:
        assert tuple(csv.DictReader(handle).fieldnames) == mod.PRIVATE_GROUND_TRUTH_FIELDS
    with (destination / "known_identity_registry.csv").open(newline="", encoding="utf-8") as handle:
        assert "original_identity" not in csv.DictReader(handle).fieldnames
