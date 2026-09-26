#!/usr/bin/env python3
"""Validate and enrich explicit species metadata for BHCA without changing splits."""

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name, relative_path):
    """Load a repo module by file path without importing apexid package __init__ (torch)."""
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_species_metadata = _load_module(
    "apexid_species_metadata_standalone",
    "apexid/data/species_metadata.py",
)
_provenance = _load_module(
    "apexid_provenance_standalone",
    "apexid/utils/provenance.py",
)
enrich_species_metadata = _species_metadata.enrich_species_metadata
DEFAULT_HAPPYWHALE_SPECIES_CSV = _species_metadata.DEFAULT_HAPPYWHALE_SPECIES_CSV
write_json_atomic = _provenance.write_json_atomic


def _write_csv_atomic(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_csv(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def prepare_species_csv(
    input_path,
    output_path=None,
    *,
    check_only=False,
    report_path=None,
    happywhale_species_csv=None,
):
    """Enrich one CSV and optionally write it atomically plus an audit report."""
    input_path = Path(input_path)
    original = pd.read_csv(input_path, low_memory=False)
    enriched, report = enrich_species_metadata(
        original,
        require_complete=True,
        happywhale_species_csv=happywhale_species_csv,
    )
    if not check_only:
        if output_path is None:
            raise ValueError("output_path is required unless check_only=True.")
        _write_csv_atomic(enriched, output_path)
    if report_path is not None:
        write_json_atomic(report_path, report.to_dict())
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source split CSV.")
    parser.add_argument("--output", type=Path, help="Enriched output CSV.")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Atomically replace --input. This never changes row membership or ordering.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and print the report without writing a CSV.",
    )
    parser.add_argument("--report", type=Path, help="Optional JSON audit-report path.")
    parser.add_argument(
        "--happywhale-species",
        type=Path,
        default=None,
        help=(
            "Official HappyWhale image/species/individual_id CSV. "
            f"Defaults to auto-detect ({DEFAULT_HAPPYWHALE_SPECIES_CSV}) when present."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.in_place and args.output is not None:
        raise SystemExit("Use either --in-place or --output, not both.")
    if args.check_only and (args.in_place or args.output is not None):
        raise SystemExit("--check-only cannot be combined with --in-place or --output.")
    output_path = args.input if args.in_place else args.output
    if not args.check_only and output_path is None:
        raise SystemExit("Provide --output, --in-place, or --check-only.")
    happywhale_csv = args.happywhale_species
    if happywhale_csv is None and DEFAULT_HAPPYWHALE_SPECIES_CSV.is_file():
        happywhale_csv = DEFAULT_HAPPYWHALE_SPECIES_CSV
    report = prepare_species_csv(
        args.input,
        output_path,
        check_only=args.check_only,
        report_path=args.report,
        happywhale_species_csv=happywhale_csv,
    )
    print(
        "BHCA species metadata valid: "
        f"rows={report.num_rows}, null_before={report.null_species_before}, "
        f"filled={report.rows_filled}, null_after={report.null_species_after}, "
        f"filled_by_dataset={report.rows_by_dataset}, "
        f"happywhale_official_filled={report.happywhale_rows_filled_from_official}"
    )


if __name__ == "__main__":
    main()
