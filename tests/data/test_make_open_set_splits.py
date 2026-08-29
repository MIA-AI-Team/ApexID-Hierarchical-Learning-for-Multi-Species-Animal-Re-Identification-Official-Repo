import csv
import json
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from apexid.data.split_v2 import (
    COMPONENT_SPLITS,
    FIELDNAMES,
    FeatureSpec,
    SplitConfig,
    annotate_dates,
    annotate_explicit_groups,
    annotate_image_hashes,
    build_features_by_index,
    choose_unknown_keys,
    explicit_source_key,
    feature_cache_fingerprint,
    group_rows_by_identity,
    make_observation_aware_splits,
    manifest_rows,
    plan_identity_methods,
    sha256_file,
    validate_feature_cache,
    validate_splits,
    validate_written_v2,
    write_v2_artifacts,
)


def _row(index, dataset="DatasetA", identity="id", date="", path=None):
    return {
        "identity": identity,
        "path": path or f"images/{dataset}/{identity}_{index}.png",
        "date": date,
        "orientation": "",
        "species": "animal",
        "dataset": dataset,
        "_idx": index,
        "_key": (dataset, identity),
        "_image_sha256": f"hash-{index}",
        "_duplicate_group_id": "",
        "_explicit_source": "",
        "_explicit_group_id": "",
        "_similarity_cluster_id": "",
    }


def _prepare(rows, enabled=()):
    annotate_dates(rows)
    annotate_explicit_groups(rows, enabled)
    return rows


def _orthogonal_features(rows):
    eye = np.eye(len(rows), dtype=np.float32)
    return {int(row["_idx"]): eye[index] for index, row in enumerate(rows)}


def _split(rows, config=None, unknown=None, features=None):
    config = config or SplitConfig(unknown_identity_ratio=0.0)
    by_key = group_rows_by_identity(rows)
    unknown = unknown or set()
    methods, _ = plan_identity_methods(by_key, unknown)
    return make_observation_aware_splits(
        rows,
        config,
        methods,
        unknown,
        features if features is not None else _orthogonal_features(rows),
    )


def test_temporal_suffix_is_chronological_and_protects_same_day():
    rows = _prepare(
        [
            _row(0, date="2020-01-01"),
            _row(1, date="2020-01-01"),
            _row(2, date="02-01-2020"),
            _row(3, date="2020-01-03"),
        ]
    )
    splits = _split(rows, SplitConfig(known_image_ratio=0.5, unknown_identity_ratio=0.0))
    train_dates = {row["_parsed_date"] for row in splits["train"]}
    val_dates = {row["_parsed_date"] for row in splits["val_known"]}
    assert max(train_dates) < min(val_dates)
    assert not train_dates & val_dates
    assert {row["_idx"] for row in rows[:2]} <= {
        row["_idx"] for row in splits["train"]
    }
    validate_splits(rows, splits)


def test_two_date_identity_keeps_one_earlier_train_group():
    rows = _prepare([_row(0, date="2024-01-01"), _row(1, date="2024-01-02")])
    splits = _split(rows, SplitConfig(known_image_ratio=0.9, unknown_identity_ratio=0.0))
    assert [row["_idx"] for row in splits["train"]] == [0]
    assert [row["_idx"] for row in splits["val_known"]] == [1]


def test_temporal_route_never_splits_cross_date_exact_must_link():
    rows = _prepare(
        [
            _row(0, date="2024-01-01"),
            _row(1, date="2024-01-02"),
            _row(2, date="2024-01-03"),
        ]
    )
    rows[0]["_image_sha256"] = rows[2]["_image_sha256"] = "same-bytes"
    rows[0]["_duplicate_group_id"] = rows[2]["_duplicate_group_id"] = "dup"
    splits = _split(rows)
    assert len(splits["train"]) == 3
    assert not splits["val_known"]
    validate_splits(rows, splits)


@pytest.mark.parametrize(
    ("dataset", "path", "expected"),
    [
        ("CTai", "x/img-id17-object-2_39.png", "CTai:17"),
        ("Giraffes", "x/original_name_3_0123456789abcdef.jpg", "Giraffes:original_name"),
        ("IPanda50", "x/36_v040_f000110_000012e43ecad67a.jpg", "IPanda50:36_v040"),
        ("IPanda50", "x/49_20150303-110300_e__1655_abc.jpg", "IPanda50:49_20150303-110300"),
        ("NDD20", "x/12345_7.jpg", "NDD20:12345"),
        ("NyalaData", "x/804_leftphoto(2)_0123456789abcdef.jpg", "NyalaData:804"),
        ("NyalaData", "x/804_rightphoto_0123456789abcdef.jpg", "NyalaData:804"),
    ],
)
def test_verified_explicit_source_parsers(dataset, path, expected):
    row = _row(0, dataset=dataset, path=path)
    assert explicit_source_key(row, {dataset}) == expected
    assert explicit_source_key(row, set()) == ""


def test_explicit_groups_are_assigned_whole():
    rows = _prepare(
        [
            _row(0, dataset="NDD20", path="1_0.jpg"),
            _row(1, dataset="NDD20", path="1_1.jpg"),
            _row(2, dataset="NDD20", path="2_0.jpg"),
            _row(3, dataset="NDD20", path="3_0.jpg"),
        ],
        enabled={"NDD20"},
    )
    splits = _split(rows, SplitConfig(known_image_ratio=0.5, unknown_identity_ratio=0.0))
    split_by_index = {
        row["_idx"]: name for name in COMPONENT_SPLITS for row in splits[name]
    }
    assert split_by_index[0] == split_by_index[1]
    validate_splits(rows, splits)


def test_similarity_and_must_link_components_stay_train():
    rows = _prepare([_row(index) for index in range(5)])
    rows[2]["_image_sha256"] = "duplicate"
    rows[3]["_image_sha256"] = "duplicate"
    rows[2]["_duplicate_group_id"] = rows[3]["_duplicate_group_id"] = "dup"
    features = np.eye(5, dtype=np.float32)
    features[1] = features[0]
    splits = _split(
        rows,
        SplitConfig(known_image_ratio=0.4, unknown_identity_ratio=0.0),
        features=build_features_by_index(rows, features),
    )
    train_indices = {row["_idx"] for row in splits["train"]}
    val_indices = {row["_idx"] for row in splits["val_known"]}
    assert {0, 1, 2, 3} <= train_indices
    assert val_indices == {4}
    assert rows[0]["_similarity_cluster_id"] == rows[1]["_similarity_cluster_id"]
    validate_splits(rows, splits)


def test_single_observation_group_is_train_only():
    rows = _prepare([_row(index) for index in range(3)])
    for row in rows:
        row["_image_sha256"] = "same"
        row["_duplicate_group_id"] = "dup"
    splits = _split(rows)
    assert len(splits["train"]) == 3
    assert not splits["val_known"]


def test_partial_missing_and_malformed_dates_use_similarity_fallback():
    rows = _prepare(
        [
            _row(0, date="2024-01-01"),
            _row(1, date=""),
            _row(2, date="01/03/2024"),
        ]
    )
    assert Counter(row["_date_format"] for row in rows) == {
        "iso": 1,
        "missing": 1,
        "malformed": 1,
    }
    methods, _ = plan_identity_methods(group_rows_by_identity(rows), set())
    assert methods[("DatasetA", "id")] == "similarity_fallback"
    splits = _split(rows)
    assert splits["train"] and splits["val_known"]
    validate_splits(rows, splits)


def test_exact_content_label_conflict_fails_with_report(tmp_path):
    image = tmp_path / "same.png"
    Image.new("RGB", (4, 4), (10, 20, 30)).save(image)
    rows = [
        _row(0, identity="one", path="same.png"),
        _row(1, identity="two", path="same.png"),
    ]
    with pytest.raises(ValueError, match="conflicting canonical labels"):
        annotate_image_hashes(rows, tmp_path)


def test_unknown_identities_are_disjoint_and_cap_respects_groups():
    rows = _prepare(
        [_row(index, identity="known") for index in range(3)]
        + [_row(index + 3, identity="unknown") for index in range(4)]
    )
    for index in (3, 4):
        rows[index]["_image_sha256"] = "unknown-duplicate"
        rows[index]["_duplicate_group_id"] = "dup-unknown"
    unknown = {("DatasetA", "unknown")}
    splits = _split(
        rows,
        SplitConfig(unknown_identity_ratio=0.0, max_unknown_images_per_identity=1),
        unknown=unknown,
    )
    train_ids = {row["identity"] for row in splits["train"]}
    unknown_ids = {row["identity"] for row in splits["val_unknown"]}
    assert not train_ids & unknown_ids
    duplicate_splits = {
        name
        for name in ("val_unknown", "val_unknown_extra")
        if any(row["_image_sha256"] == "unknown-duplicate" for row in splits[name])
    }
    assert len(duplicate_splits) == 1
    validate_splits(rows, splits)


def test_singletons_and_new_whale_route_to_unlabeled_ood():
    rows = _prepare(
        [
            _row(0, identity="singleton"),
            _row(1, dataset="humpback-whale-identification", identity="new_whale"),
            _row(2, dataset="humpback-whale-identification", identity="new_whale"),
        ]
    )
    splits = _split(rows)
    assert {row["_idx"] for row in splits["unlabeled_ood"]} == {0, 1, 2}
    validate_splits(rows, splits)


def test_outputs_are_deterministic_and_complete(tmp_path):
    rows = _prepare([_row(index) for index in range(5)])
    config = SplitConfig(seed=17, unknown_identity_ratio=0.0)
    splits = _split(rows, config)
    summary = {"integrity": validate_splits(rows, splits)}
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_v2_artifacts(first, splits, manifest_rows(rows, config), dict(summary))
    write_v2_artifacts(second, splits, manifest_rows(rows, config), dict(summary))
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows({name: row[name] for name in FIELDNAMES} for row in rows)
    assert validate_written_v2(first, metadata)["passed"]
    with pytest.raises(FileExistsError):
        write_v2_artifacts(first, splits, manifest_rows(rows, config), dict(summary))


def test_unrelated_source_rows_do_not_reshuffle_an_identity():
    original = _prepare([_row(index, identity="stable") for index in range(5)])
    original_splits = _split(original, SplitConfig(seed=123, unknown_identity_ratio=0.0))
    original_assignment = {
        row["path"]: (name, row["_observation_group_id"])
        for name in COMPONENT_SPLITS
        for row in original_splits[name]
    }

    shifted = [_row(0, identity="unrelated")]
    shifted.extend(
        _row(index + 100, identity="stable", path=original[index]["path"])
        for index in range(5)
    )
    shifted = _prepare(shifted)
    shifted_splits = _split(shifted, SplitConfig(seed=123, unknown_identity_ratio=0.0))
    shifted_assignment = {
        row["path"]: (name, row["_observation_group_id"])
        for name in COMPONENT_SPLITS
        for row in shifted_splits[name]
        if row["identity"] == "stable"
    }
    assert shifted_assignment == original_assignment


def test_feature_cache_rejects_stale_and_incomplete_inputs(tmp_path):
    rows = _prepare([_row(0), _row(1)])
    feature = FeatureSpec(dimension=3)
    metadata_hash = "metadata-sha"
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(ValueError, match="Incomplete"):
        validate_feature_cache(cache, rows, metadata_hash, feature)

    features_path = cache / "features.npy"
    np.save(features_path, np.eye(2, 3, dtype=np.float32), allow_pickle=False)
    rows_path = cache / "feature_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_index", "path", "image_sha256"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_index": row["_idx"],
                    "path": row["path"],
                    "image_sha256": row["_image_sha256"],
                }
            )
    metadata = {
        "fingerprint": feature_cache_fingerprint(rows, metadata_hash, feature),
        "features_sha256": sha256_file(features_path),
        "rows_sha256": sha256_file(rows_path),
    }
    (cache / "cache_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert validate_feature_cache(cache, rows, metadata_hash, feature).shape == (2, 3)
    with pytest.raises(ValueError, match="Stale"):
        validate_feature_cache(cache, rows, "changed", feature)


def test_unknown_selection_is_seeded_and_dataset_wise():
    rows = _prepare(
        [
            _row(index * 2 + offset, identity=f"id-{index}")
            for index in range(12)
            for offset in range(2)
        ]
    )
    by_key = group_rows_by_identity(rows)
    _, first = choose_unknown_keys(by_key, 42, 0.125)
    _, second = choose_unknown_keys(by_key, 42, 0.125)
    assert first == second
    assert len(first) == 2
