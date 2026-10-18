import json

import pytest
import yaml

from graphslm_ids.offline.federated.cli import headline, main


def test_cli_runs_one_experiment(smoke_cfg, tmp_path, capsys):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(smoke_cfg("cli")), encoding="utf-8")
    code = main(
        ["--config", str(path), "--rounds", "1", "--set", "attack.name=sign_flip", "--set", "attack.malicious_ratio=0.34"]
    )
    assert code == 0
    assert '"attack": "sign_flip"' in capsys.readouterr().out
    summary = json.loads((tmp_path / "cli" / "fl_summary.json").read_text(encoding="utf-8"))
    assert summary["rounds"] == 1


def test_cli_rejects_malformed_set(smoke_cfg, tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(smoke_cfg("bad")), encoding="utf-8")
    with pytest.raises(ValueError):
        main(["--config", str(path), "--set", "attack.name"])


def test_headline_reports_model_not_screen():
    summary = {
        "experiment": "e",
        "mode": "fl",
        "model": "hgt",
        "attack": "none",
        "aggregator": "fedavg",
        "final": {"test": {"macro_f1": 0.5, "clean": {"macro_f1": 0.6}}},
        "detection": {"tp": 1},
    }
    head = headline(summary)
    assert head["model"] == "hgt" and "screen" not in head
    assert head["test_macro_f1"] == 0.5 and head["test_macro_f1_clean"] == 0.6 and head["detection"] == {"tp": 1}
    assert headline({})["test_macro_f1"] is None
