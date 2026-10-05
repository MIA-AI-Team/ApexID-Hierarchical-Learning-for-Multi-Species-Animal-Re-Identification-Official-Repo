"""Regression coverage for whale-category migration without importing torch."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from scripts.prepare_bhca_species_metadata import (
    REPO_ROOT,
    _species_metadata,
    prepare_species_csv,
)


class WhaleCategoryMigrationTests(unittest.TestCase):
    def test_existing_aliases_missing_labels_and_idempotence(self):
        species = [
            "humpback-whale", "humpback_whale", "humpback whale",
            " HUMPBACK-WHALE ", "Whale", "whale", None,
            "dolphin", "giraffe", "zebra", "whale-shark",
        ]
        original = pd.DataFrame({
            "identity": [f"humpback-whale-id-{i}" for i in range(len(species))],
            "path": [f"humpback-whale/images/{i}.jpg" for i in range(len(species))],
            "species": species,
            "dataset": ["humpback-whale-identification"] * 7 + ["Other"] * 4,
            "observation_id": [f"observation-{i}" for i in range(len(species))],
        }, index=range(10, 10 + len(species)))
        before = original.copy(deep=True)

        enriched, report = _species_metadata.enrich_species_metadata(original)

        self.assertEqual(enriched["species"].tolist(), ["whale"] * 7 + species[7:])
        pd.testing.assert_frame_equal(original, before)
        pd.testing.assert_frame_equal(
            enriched.drop(columns="species"), original.drop(columns="species")
        )
        self.assertEqual(report.rows_filled, 1)
        self.assertEqual(report.rows_normalized, 5)
        self.assertEqual(
            report.rows_normalized_by_dataset, {"humpback-whale-identification": 5}
        )
        repeated, second_report = _species_metadata.enrich_species_metadata(enriched)
        pd.testing.assert_frame_equal(repeated, enriched)
        self.assertEqual(second_report.rows_filled, 0)
        self.assertEqual(second_report.rows_normalized, 0)

    def test_categories_merge_across_datasets_before_building_hierarchy(self):
        with tempfile.TemporaryDirectory() as directory:
            official = Path(directory) / "official.csv"
            pd.DataFrame({
                "image": ["whale.jpg", "dolphin.jpg"],
                "species": ["humpback_whale", "bottlenose_dolphin"],
                "individual_id": ["happy-whale", "happy-dolphin"],
            }).to_csv(official, index=False)
            original = pd.DataFrame({
                "identity": ["humpback", "happy-whale", "happy-dolphin", "other-whale"],
                "path": ["old.jpg", "whale.jpg", "dolphin.jpg", "other.jpg"],
                "species": ["humpback-whale", "whale", None, "Humpback Whale"],
                "dataset": [
                    "humpback-whale-identification", "HappyWhale", "HappyWhale", "Other"
                ],
            })
            enriched, report = _species_metadata.enrich_species_metadata(
                original, happywhale_species_csv=official
            )
            hierarchy = _species_metadata.build_training_hierarchy(
                enriched, {label: i for i, label in enumerate(enriched["identity"])}
            )

        self.assertEqual(enriched["species"].tolist(), ["whale", "whale", "dolphin", "whale"])
        self.assertEqual(hierarchy.species_names, ["dolphin", "whale"])
        self.assertEqual(hierarchy.identities_per_species, [1, 3])
        self.assertEqual(hierarchy.class_to_species, [1, 1, 0, 1])
        self.assertEqual(report.happywhale_rows_filled_from_official, 1)
        self.assertEqual(report.rows_normalized, 2)

    def test_unrelated_conflicts_still_fail(self):
        for label in ["dolphin", "giraffe", "whale-shark"]:
            with self.subTest(label=label):
                frame = pd.DataFrame({
                    "dataset": ["humpback-whale-identification"],
                    "species": [label],
                })
                with self.assertRaisesRegex(ValueError, "conflicts with approved species"):
                    _species_metadata.enrich_species_metadata(frame)

    def test_custom_columns_and_legacy_mapping_override(self):
        frame = pd.DataFrame({
            "source": ["CustomWhales", "CustomWhales"],
            "category": ["humpback_whale", None],
            "animal": ["a", "b"],
        })
        enriched, report = _species_metadata.enrich_species_metadata(
            frame,
            dataset_column="source",
            species_column="category",
            identity_column="animal",
            dataset_to_species={"CustomWhales": "humpback-whale"},
        )
        self.assertEqual(enriched["category"].tolist(), ["whale", "whale"])
        self.assertEqual(report.canonical_dataset_species["CustomWhales"], "whale")
        self.assertEqual(report.rows_filled, 1)
        self.assertEqual(report.rows_normalized, 1)

    def test_output_copy_preserves_source_and_reports_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, audit = root / "input.csv", root / "output.csv", root / "audit.json"
            original = pd.DataFrame({
                "identity": ["humpback-whale-id", "whale-id"],
                "path": ["humpback-whale/1.jpg", "humpback-whale/2.jpg"],
                "species": ["humpback-whale", None],
                "dataset": ["humpback-whale-identification"] * 2,
            })
            original.to_csv(source, index=False)
            before = source.read_bytes()
            report = prepare_species_csv(source, output, report_path=audit)

            self.assertEqual(source.read_bytes(), before)
            enriched = pd.read_csv(output)
            self.assertEqual(enriched["species"].tolist(), ["whale", "whale"])
            pd.testing.assert_frame_equal(
                enriched.drop(columns="species"), original.drop(columns="species")
            )
            self.assertEqual(report.rows_filled, 1)
            self.assertEqual(report.rows_normalized, 1)
            self.assertEqual(json.loads(audit.read_text())["rows_normalized"], 1)

    def test_cli_check_only_in_place_and_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, audit = root / "input.csv", root / "audit.json"
            source.write_text(
                "identity,path,species,dataset\n"
                "humpback-id,humpback-whale/a.jpg,humpback-whale,humpback-whale-identification\n"
            )
            before = source.read_bytes()
            command = [
                sys.executable, str(REPO_ROOT / "scripts/prepare_bhca_species_metadata.py"),
                "--input", str(source), "--report", str(audit),
            ]
            checked = subprocess.run(
                command + ["--check-only"], check=True, capture_output=True, text=True
            )
            self.assertEqual(source.read_bytes(), before)
            self.assertIn("normalized=1", checked.stdout)
            subprocess.run(command + ["--in-place"], check=True, capture_output=True, text=True)
            self.assertEqual(pd.read_csv(source)["species"].tolist(), ["whale"])
            self.assertEqual(json.loads(audit.read_text())["rows_normalized"], 1)
            migrated_bytes = source.read_bytes()
            repeated = subprocess.run(
                command + ["--in-place"], check=True, capture_output=True, text=True
            )
            self.assertEqual(source.read_bytes(), migrated_bytes)
            self.assertIn("normalized=0", repeated.stdout)


if __name__ == "__main__":
    unittest.main()
