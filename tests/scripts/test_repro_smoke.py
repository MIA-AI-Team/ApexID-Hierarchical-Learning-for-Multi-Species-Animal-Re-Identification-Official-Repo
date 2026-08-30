import json
import os
import subprocess
import sys
from pathlib import Path


def test_offline_reproduction_smoke(tmp_path):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "smoke.json"
    environment = os.environ.copy()
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "NO_ALBUMENTATIONS_UPDATE": "1"})
    completed = subprocess.run(
        [sys.executable, "scripts/repro_smoke.py", "--output", str(output)],
        cwd=root, env=environment, text=True, capture_output=True, check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert "PASS" in completed.stdout
    assert result["status"] == "pass"
    assert result["checkpoint_round_trip_equal"] is True
    assert result["online_assignments"] == ["NEW_000001", "NEW_000001", "NEW_000002"]
