from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import torch
import yaml


GITHUB_REPO_URL = ""
GITHUB_BRANCH = ""

WORK_DIR = Path("/kaggle/working/nt114_hgt_work")
GRAPH_NPZ_NAME = "graph_artifact_3tier_t082_k5.npz"
GRAPH_META_NAME = "graph_artifact_3tier_t082_k5.meta.json"
RESULT_ZIP = Path("/kaggle/working/hgt_training_results_kaggle.zip")

RUNS = [
    {
        "name": "baseline_t082_k5_l3_d01",
        "config": "configs/hgt_t082_k5_l3_d01.yaml",
        "source_name": "Project baseline t082_k5_l3_d01",
        "citation_hint": "Local baseline selected by threshold experiment t082_k5",
    },
    {
        "name": "xgnid_dual_modal_l1_h32_h4",
        "config": "configs/hgt_paper_variants/hgt_t082_k5_xgnid_dual_modal_l1_h32_h4.yaml",
    },
    {
        "name": "one2_iov_l1_h64_h2",
        "config": "configs/hgt_paper_variants/hgt_t082_k5_one2_iov_l1_h64_h2.yaml",
    },
    {
        "name": "relgt_multi_token_l3_h128_h8",
        "config": "configs/hgt_paper_variants/hgt_t082_k5_relgt_multi_token_l3_h128_h8.yaml",
    },
    {
        "name": "gatransformer_deep_l6_h256_h8",
        "config": "configs/hgt_paper_variants/hgt_t082_k5_gatransformer_deep_l6_h256_h8.yaml",
    },
    {
        "name": "ahgt_dfd_funnel_l3_h128_h4",
        "config": "configs/hgt_paper_variants/hgt_t082_k5_ahgt_dfd_funnel_l3_h128_h4.yaml",
    },
    {
        "name": "dlg_ids_sparse_l2_h128_h4",
        "config": "configs/hgt_paper_variants/hgt_t082_k5_dlg_ids_sparse_l2_h128_h4.yaml",
    },
]


def is_repo_root(path: Path) -> bool:
    return (
        (path / "src" / "graphslm_ids").exists()
        and (path / "configs" / "hgt_t082_k5_l3_d01.yaml").exists()
        and (path / "pyproject.toml").exists()
    )


def prepare_repo() -> None:
    if is_repo_root(WORK_DIR):
        print("Work dir exists:", WORK_DIR)
        return

    if GITHUB_REPO_URL.strip():
        if WORK_DIR.exists():
            shutil.rmtree(WORK_DIR)
        cmd = ["git", "clone"]
        if GITHUB_BRANCH.strip():
            cmd += ["--branch", GITHUB_BRANCH]
        cmd += [GITHUB_REPO_URL, str(WORK_DIR)]
        print("Cloning:", " ".join(cmd))
        subprocess.check_call(cmd)
        return

    candidates: list[Path] = []
    for root in Path("/kaggle/input").glob("*"):
        if is_repo_root(root):
            candidates.append(root)
        for config_path in root.rglob("configs/hgt_t082_k5_l3_d01.yaml"):
            candidate = config_path.parents[1]
            if is_repo_root(candidate):
                candidates.append(candidate)
    if not candidates:
        raise RuntimeError("Repo not found. Set GITHUB_REPO_URL or add repo as Kaggle Dataset.")

    source = candidates[0]
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    print("Copy repo:", source, "->", WORK_DIR)
    shutil.copytree(source, WORK_DIR, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))


def find_input_file(name: str) -> Path:
    direct = WORK_DIR / "data" / "processed" / name
    if direct.exists():
        return direct
    matches = sorted(Path("/kaggle/input").rglob(name))
    if not matches:
        raise FileNotFoundError(f"Missing {name}. Add it to a Kaggle Dataset.")
    return matches[0]


def prepare_graph_files() -> tuple[Path, Path]:
    processed = WORK_DIR / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    graph_npz = find_input_file(GRAPH_NPZ_NAME)
    graph_meta = find_input_file(GRAPH_META_NAME)
    target_npz = processed / GRAPH_NPZ_NAME
    target_meta = processed / GRAPH_META_NAME
    if graph_npz.resolve() != target_npz.resolve():
        shutil.copy2(graph_npz, target_npz)
    if graph_meta.resolve() != target_meta.resolve():
        shutil.copy2(graph_meta, target_meta)
    return target_npz, target_meta


def install_repo() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."], cwd=WORK_DIR)


def build_graph_store(graph_npz: Path, graph_meta: Path) -> None:
    manifest = WORK_DIR / "data" / "graph_store_v1" / "manifest.json"
    if manifest.exists():
        print("Graph store exists:", manifest)
        return
    cmd = [
        sys.executable,
        "-m",
        "graphslm_ids.offline_path.training.on_disk_graph_store",
        "--graph-npz",
        str(graph_npz),
        "--graph-meta-json",
        str(graph_meta),
        "--output-root",
        str(WORK_DIR / "data" / "graph_store_v1"),
    ]
    print("Convert graph store:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=WORK_DIR)


def config_summary_path(config_path: str) -> Path:
    with (WORK_DIR / config_path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return WORK_DIR / cfg["train"]["output_dir"] / "training_summary.json"


def run_training(run: dict[str, str], device: str) -> tuple[int, Path]:
    log_dir = WORK_DIR / "outputs" / "hgt_kaggle_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if device == "cuda" else "_cpu_fallback"
    log_path = log_dir / f"{run['name']}{suffix}.log"
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "graphslm_ids.offline_path.training.train_hgt_flow_classifier",
        "--config",
        run["config"],
        "--device",
        device,
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    print(f"\n=== TRAIN {run['name']} on {device.upper()} ===")
    print("config:", run["config"])
    print("log:", log_path)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd,
            cwd=WORK_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    return code, log_path


def log_has_cuda_oom(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    return any(
        marker in text
        for marker in (
            "CUDA out of memory",
            "torch.OutOfMemoryError",
            "RuntimeError: CUDA error: out of memory",
        )
    )


def train_all() -> None:
    for run in RUNS:
        summary_path = config_summary_path(run["config"])
        if summary_path.exists():
            print(f"\n=== SKIP {run['name']} ===")
            print("summary exists:", summary_path)
            continue
        code, log_path = run_training(run, "cuda")
        if code == 0:
            continue
        if log_has_cuda_oom(log_path):
            print(f"\n[WARN] {run['name']} ran out of GPU memory. Retrying on CPU...")
            code, cpu_log_path = run_training(run, "cpu")
            if code == 0:
                continue
            raise RuntimeError(f"CPU fallback failed: {run['config']}. See {cpu_log_path}")
        raise RuntimeError(f"Run failed: {run['config']}. See {log_path}")


def build_comparison() -> None:
    rows: list[dict[str, object]] = []
    for run in RUNS:
        summary_path = config_summary_path(run["config"])
        if not summary_path.exists():
            print("Missing summary, skip:", run["name"], summary_path)
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        cfg = data["config"]
        exp = cfg.get("experiment", {})
        model = cfg["model"]
        train = cfg["train"]
        best_val = data.get("best_val_metrics", {})
        best_test = data.get("best_test_metrics", {})
        rows.append(
            {
                "run_name": run["name"],
                "run_dir": str(summary_path.parent.relative_to(WORK_DIR)),
                "source_name": exp.get("source_name", run.get("source_name", "")),
                "citation_hint": exp.get("citation_hint", run.get("citation_hint", "")),
                "hidden_dim": model["hidden_dim"],
                "num_layers": model["num_layers"],
                "num_heads": model["num_heads"],
                "dropout": model.get("dropout"),
                "batch_mode": train.get("batch_mode"),
                "best_epoch": data.get("best_epoch"),
                "best_score": data.get("best_score"),
                "val_macro_f1": best_val.get("macro_f1"),
                "test_macro_f1": best_test.get("macro_f1"),
                "test_accuracy": best_test.get("accuracy"),
                "device": data.get("device"),
            }
        )

    out_dir = WORK_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "hgt_kaggle_comparison.csv"
    out_md = out_dir / "hgt_kaggle_comparison.md"
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        out_md.write_text(_markdown_table(rows), encoding="utf-8")
    print("Comparison:", out_csv)


def _markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def bundle_results() -> None:
    if RESULT_ZIP.exists():
        RESULT_ZIP.unlink()
    include_roots = [
        WORK_DIR / "outputs",
        WORK_DIR / "configs" / "hgt_t082_k5_l3_d01.yaml",
        WORK_DIR / "configs" / "hgt_paper_variants",
    ]
    with zipfile.ZipFile(RESULT_ZIP, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for root in include_roots:
            if not root.exists():
                continue
            if root.is_file():
                zf.write(root, root.relative_to(WORK_DIR))
                continue
            for path in root.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(WORK_DIR))
    print("Bundle:", RESULT_ZIP)
    print("Size MB:", round(RESULT_ZIP.stat().st_size / 1024 / 1024, 2))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Enable Kaggle GPU before running this train file.")
    prepare_repo()
    os.chdir(WORK_DIR)
    graph_npz, graph_meta = prepare_graph_files()
    install_repo()
    build_graph_store(graph_npz, graph_meta)
    train_all()
    build_comparison()
    bundle_results()


if __name__ == "__main__":
    main()
