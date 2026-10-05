import pandas as pd
import pytest

from apexid.data.species_metadata import (
    BHCA_DATASET_TO_SPECIES,
    canonicalize_happywhale_species,
    enrich_species_metadata,
)
from scripts.prepare_bhca_species_metadata import prepare_species_csv


def _metadata_frame():
    return pd.DataFrame(
        {
            "identity": ["l1", "s1", "h1", "g1", "z1"],
            "path": [f"image_{index}.jpg" for index in range(5)],
            "species": ["lynx", pd.NA, pd.NA, "giraffe", "zebra"],
            "dataset": [
                "CzechLynx",
                "SalamanderID2025",
                "humpback-whale-identification",
                "GiraffeZebraID",
                "GiraffeZebraID",
            ],
        }
    )


def _happywhale_official(tmp_path):
    path = tmp_path / "happywhale_official.csv"
    pd.DataFrame(
        {
            "image": ["hw_dolphin.jpg", "hw_whale.jpg", "hw_typo.jpg"],
            "species": ["bottlenose_dolphin", "humpback_whale", "bottlenose_dolpin"],
            "individual_id": ["d1", "w1", "d2"],
        }
    ).to_csv(path, index=False)
    return path


def test_canonicalize_happywhale_species_is_binary():
    assert canonicalize_happywhale_species("bottlenose_dolphin") == "dolphin"
    assert canonicalize_happywhale_species("bottlenose_dolpin") == "dolphin"
    assert canonicalize_happywhale_species("killer_whale") == "whale"
    assert canonicalize_happywhale_species("beluga") == "whale"
    assert canonicalize_happywhale_species("humpback_whale") == "whale"


def test_enrichment_fills_approved_datasets_and_preserves_other_metadata():
    original = _metadata_frame()

    enriched, report = enrich_species_metadata(original)

    assert enriched["species"].tolist() == [
        "lynx",
        "salamander",
        "whale",
        "giraffe",
        "zebra",
    ]
    assert not enriched["species"].isna().any()
    pd.testing.assert_frame_equal(
        enriched.drop(columns="species"),
        original.drop(columns="species"),
    )
    assert enriched.index.equals(original.index)
    assert report.num_rows == len(original)
    assert report.null_species_before == 2
    assert report.rows_filled == 2
    assert report.null_species_after == 0
    assert report.rows_by_dataset == {
        "CzechLynx": 0,
        "SalamanderID2025": 1,
        "humpback-whale-identification": 1,
    }
    assert report.canonical_dataset_species == BHCA_DATASET_TO_SPECIES
    assert "HappyWhale" not in BHCA_DATASET_TO_SPECIES


def test_enrichment_fills_happywhale_as_whale_or_dolphin(tmp_path):
    official = _happywhale_official(tmp_path)
    frame = pd.DataFrame(
        {
            "identity": ["d1", "w1", "d2"],
            "path": [
                "images/HappyWhale/train/hw_dolphin.jpg",
                "images/HappyWhale/train/hw_whale.jpg",
                "images/HappyWhale/train/hw_typo.jpg",
            ],
            "species": [pd.NA, pd.NA, pd.NA],
            "dataset": ["HappyWhale", "HappyWhale", "HappyWhale"],
        }
    )

    enriched, report = enrich_species_metadata(
        frame, happywhale_species_csv=official
    )

    assert enriched["species"].tolist() == ["dolphin", "whale", "dolphin"]
    assert report.happywhale_rows_filled_from_official == 3
    assert report.rows_by_dataset["HappyWhale"] == 3


def test_enrichment_rejects_happywhale_conflict_with_official(tmp_path):
    official = _happywhale_official(tmp_path)
    frame = pd.DataFrame(
        {
            "identity": ["d1"],
            "path": ["images/HappyWhale/train/hw_dolphin.jpg"],
            "species": ["whale"],
            "dataset": ["HappyWhale"],
        }
    )

    with pytest.raises(ValueError, match="conflicts with official"):
        enrich_species_metadata(frame, happywhale_species_csv=official)


def test_enrichment_rejects_dataset_level_happywhale_override():
    frame = _metadata_frame()
    with pytest.raises(ValueError, match="must not use a single dataset-level"):
        enrich_species_metadata(frame, dataset_to_species={"HappyWhale": "whale"})


def test_enrichment_rejects_conflicting_approved_species():
    frame = _metadata_frame()
    frame.loc[frame["dataset"].eq("CzechLynx"), "species"] = "dolphin"

    with pytest.raises(ValueError, match="conflicts with approved species"):
        enrich_species_metadata(frame)


def test_enrichment_rejects_unresolved_null_species():
    frame = _metadata_frame()
    frame.loc[len(frame)] = {
        "identity": "unknown1",
        "path": "unknown.jpg",
        "species": pd.NA,
        "dataset": "UnmappedDataset",
    }

    with pytest.raises(ValueError, match="Unresolved rows by dataset: UnmappedDataset=1"):
        enrich_species_metadata(frame)


def test_prepare_species_csv_writes_enriched_copy_without_mutating_input(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    report_path = tmp_path / "report.json"
    original = _metadata_frame()
    original.to_csv(input_path, index=False)
    source_before = pd.read_csv(input_path)

    report = prepare_species_csv(
        input_path,
        output_path,
        report_path=report_path,
    )

    source_after = pd.read_csv(input_path)
    enriched = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(source_after, source_before)
    assert source_after["species"].isna().sum() == 2
    assert enriched["species"].isna().sum() == 0
    assert enriched["identity"].tolist() == original["identity"].tolist()
    assert report.rows_filled == 2
    assert report_path.exists()


def test_prepare_species_csv_check_only_does_not_write(tmp_path):
    input_path = tmp_path / "input.csv"
    _metadata_frame().to_csv(input_path, index=False)

    report = prepare_species_csv(input_path, check_only=True)

    assert report.rows_filled == 2
    assert list(tmp_path.iterdir()) == [input_path]
