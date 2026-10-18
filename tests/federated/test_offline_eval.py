import json

import pytest
import torch

from graphslm_ids.offline.federated.offline_eval import load_round_deltas, main, replay, save_round_deltas
from graphslm_ids.offline.federated.simulation import run_experiment


def test_float16_overflow_falls_back_to_float32(tmp_path):
    used = save_round_deltas(tmp_path, 1, [0, 1], torch.tensor([[1e6, 0.5], [0.1, 0.2]]), torch.tensor([1.0, 1.0]), [0], "float16")
    assert used == "float32"
    rec = load_round_deltas(tmp_path / "round_001.pt")
    assert bool(torch.isfinite(rec["deltas"]).all()) and float(rec["deltas"][0, 0]) == 1e6
    assert save_round_deltas(tmp_path, 2, [0], torch.tensor([[0.5, 0.25]]), torch.tensor([1.0]), [], "float16") == "float16"


def _live(smoke_cfg, tmp_path):
    cfg = smoke_cfg(
        "live",
        **{
            "attack.name": "sign_flip",
            "attack.malicious_ratio": 0.34,
            "defense.aggregator": {"name": "raguard", "params": {}},
            "output.save_deltas": True,
            "output.save_deltas_dtype": "float32",
            "federated.rounds": 3,
        },
    )
    run_experiment(cfg)
    return tmp_path / "live"


def test_replay_reproduces_live_decisions(smoke_cfg, tmp_path):
    run = _live(smoke_cfg, tmp_path)
    live = [json.loads(line) for line in (run / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    out = replay(run, {"name": "raguard", "params": {}})
    assert [r["round"] for r in out["rounds"]] == [1, 2, 3]
    for rec, rep in zip(live, out["rounds"]):
        assert {int(d["client_id"]): d["decision"] for d in rec["decisions"]} == {int(k): v for k, v in rep["decisions"].items()}
    assert out["detection"]["tp"] + out["detection"]["fn"] == 3


def test_offline_eval_cli_scores_several_defenses(smoke_cfg, tmp_path):
    run = _live(smoke_cfg, tmp_path)
    assert main(["--run", str(run), "--defense", "median", "--defense", "raguard"]) == 0
    result = json.loads((run / "offline_eval.json").read_text(encoding="utf-8"))
    assert set(result) == {"median", "raguard"} and result["median"]["detection"]["tpr"] is not None
    with pytest.raises(ValueError):
        replay(run, {"name": "bogus"})
    with pytest.raises(ValueError, match="krum"):
        main(["--run", str(run), "--defense", "multi_krum", "--params-json", '{"krum": {"f": 1}}'])


def test_bad_deltas_dtype_rejected(smoke_cfg):
    cfg = smoke_cfg("bad", **{"output.save_deltas_dtype": "bfloat16"})
    with pytest.raises(ValueError):
        run_experiment(cfg)
