import pytest

from apexid.utils.config import expand_config_environment, load_config


def test_environment_fallbacks_and_recursive_overrides(monkeypatch):
    monkeypatch.delenv("APEXID_DATA_ROOT", raising=False)
    config = {
        "path": "${APEXID_DATA_ROOT:-ApexID_Dataset}/images",
        "nested": ["${APEXID_OUTPUT_ROOT:-outputs}", {"value": "${RUN_NAME}"}],
    }
    monkeypatch.setenv("RUN_NAME", "experiment")
    monkeypatch.setenv("APEXID_OUTPUT_ROOT", "/work/results")
    expanded = load_config(config=config)
    assert expanded["path"] == "ApexID_Dataset/images"
    assert expanded["nested"] == ["/work/results", {"value": "experiment"}]


def test_environment_expansion_is_recursive_and_does_not_mutate_input():
    source = {"path": "${ROOT}", "tuple": ("${MISSING:-fallback}",)}
    expanded = expand_config_environment(source, {"ROOT": "${SECOND}", "SECOND": "/data"})
    assert expanded == {"path": "/data", "tuple": ("fallback",)}
    assert source["path"] == "${ROOT}"


def test_unresolved_environment_variable_names_the_problem():
    with pytest.raises(ValueError, match="UNSET_APEXID_VARIABLE"):
        expand_config_environment({"path": "${UNSET_APEXID_VARIABLE}"}, {})


def test_empty_environment_value_uses_fallback():
    assert expand_config_environment("${ROOT:-relative}/file", {"ROOT": ""}) == "relative/file"


def test_yaml_inheritance_merges_mappings_before_expanding_environment(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "data:\n  path: ${UNSET_APEXID_VARIABLE}\n  size: [440, 440]\nseed: 42\n",
        encoding="utf-8",
    )
    nested = tmp_path / "nested"
    nested.mkdir()
    config_path = nested / "experiment.yaml"
    config_path.write_text(
        "extends: ../base.yaml\ndata:\n  path: images\n  size: [384]\n",
        encoding="utf-8",
    )

    assert load_config(config_path=config_path) == {"data": {"path": "images", "size": [384]}, "seed": 42}


def test_yaml_inheritance_reports_cycles(tmp_path):
    first = tmp_path / "first.yaml"
    first.write_text("extends: second.yaml\n", encoding="utf-8")
    (tmp_path / "second.yaml").write_text("extends: first.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Circular config inheritance: .*first.yaml.*second.yaml.*first.yaml"):
        load_config(config_path=first)
