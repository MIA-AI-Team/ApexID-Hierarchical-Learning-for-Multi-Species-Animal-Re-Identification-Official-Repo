import json
from pathlib import Path

import pytest

from apexid.utils.provenance import (
    artifact_hashes,
    capture_environment,
    capture_git_state,
    require_clean_git,
    sha256_file,
    validate_normalization_stats,
    write_json_atomic,
    write_yaml_atomic,
)


def test_atomic_writers_and_artifact_hashes(tmp_path):
    json_path = tmp_path / "record.json"
    yaml_path = tmp_path / "record.yaml"
    write_json_atomic(json_path, {"path": Path("relative"), "value": 3})
    write_yaml_atomic(yaml_path, {"items": [1, 2]})
    assert json.loads(json_path.read_text())["path"] == "relative"
    records = artifact_hashes({"json": json_path, "missing": tmp_path / "missing"})
    assert records["json"]["sha256"] == sha256_file(json_path)
    assert records["missing"]["exists"] is False
    assert not list(tmp_path.glob(".*.tmp"))


def test_environment_capture_is_cpu_safe():
    record = capture_environment()
    assert record["python"]["version"]
    assert isinstance(record["cuda"]["available"], bool)
    assert isinstance(record["dependencies"], dict)


def test_git_capture_and_strict_modes(tmp_path):
    unavailable = capture_git_state(tmp_path)
    assert unavailable["available"] is False
    with pytest.raises(RuntimeError, match="unverifiable"):
        require_clean_git(unavailable)
    with pytest.raises(RuntimeError, match="dirty"):
        require_clean_git({"available": True, "dirty": True, "changed_paths": ["file.py"]})
    require_clean_git({"available": True, "dirty": False})


def test_normalization_stats_hash_validation(tmp_path):
    train = tmp_path / "train.csv"
    train.write_text("path,identity\na.jpg,A\n", encoding="utf-8")
    stats = tmp_path / "normalization_stats.json"
    write_json_atomic(stats, {"train_csv_sha256": sha256_file(train), "mean": [0.1] * 3, "std": [0.2] * 3})
    assert validate_normalization_stats(stats, train_csv=train)["mean"] == [0.1] * 3
    train.write_text("path,identity\nb.jpg,B\n", encoding="utf-8")
    with pytest.raises(ValueError, match="train CSV hash"):
        validate_normalization_stats(stats, train_csv=train)
