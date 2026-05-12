"""Kaggle notebook — Retrain student 1D-CNN từ pre-extracted dataset (offline).

Cách dùng
---------
Upload 3 Kaggle dataset trước khi chạy notebook:

  1. payload-dataset/
       payload_256.npy      ← kết quả bước extract_payload_dataset
       metadata.csv

  2. mitre-knowledge/       ← ít nhất 1 trong 2 lựa chọn:
       enterprise-attack.json              (script tự parse)
       HOẶC:
       mitre_techniques.csv
       mitre_tactics.csv
       mitre_technique_tactic_edges.csv
       mitre_techniques_embeddings.npy     (tuỳ chọn, bỏ qua bước embed MITRE)

  3. securebert-model/      ← thư mục model HuggingFace (offline)
       config.json
       tokenizer_config.json
       tokenizer.json
       vocab.json
       merges.txt
       pytorch_model.bin  HOẶC  model.safetensors

Pipeline khi chạy
-----------------
  [SKIP] Trích xuất PCAP    ← dùng file đã upload
  [RUN]  build_teacher_targets        (SecureBERT → teacher_targets.npy)
  [RUN]  train_student_cnn            (retrain từ đầu, luôn chạy)
  [RUN]  export_student_embeddings
  [RUN/SKIP] prepare_mitre_knowledge_base
  [RUN/SKIP] build_mitre_technique_embeddings
  [RUN]  build_three_tier_graph_artifact
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]


# ── helpers ────────────────────────────────────────────────────────────────────

def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n$", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd or REPO_DIR), check=True)


def find_input_file(input_root: Path, name: str) -> Path | None:
    matches = sorted(input_root.glob(f"**/{name}"))
    return matches[0] if matches else None


def copy_input_artifact(input_root: Path, name: str, dst: Path) -> bool:
    """Copy file từ input_root vào dst nếu chưa có. Trả về True nếu copy thành công."""
    src = find_input_file(input_root, name)
    if src is None or dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[COPY] {name}: {src} → {dst}", flush=True)
    shutil.copy2(src, dst)
    return True


def _npy_shape(path: Path) -> str:
    try:
        import numpy as np
        arr = np.load(str(path), mmap_mode="r")
        return str(arr.shape)
    except Exception:
        return "?"


def is_huggingface_model_dir(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    tokenizer_markers = {
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "spiece.model",
        "merges.txt",
    }
    return any((path / marker).is_file() for marker in tokenizer_markers)


def find_embedding_model_dirs(roots: list[Path]) -> list[Path]:
    found: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for config_path in root.glob("**/config.json"):
            model_dir = config_path.parent
            if is_huggingface_model_dir(model_dir):
                found[str(model_dir.resolve())] = model_dir
    return sorted(found.values(), key=lambda p: str(p))


def resolve_embedding_model(args: argparse.Namespace, roots: list[Path]) -> str:
    """Trả về đường dẫn local tới SecureBERT hoặc HuggingFace model ID."""
    if args.embedding_model_path.strip():
        model_path = Path(args.embedding_model_path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Embedding model path không tồn tại: {model_path}")
        if not is_huggingface_model_dir(model_path):
            raise RuntimeError(
                f"Không phải thư mục HuggingFace hợp lệ (thiếu config/tokenizer): {model_path}"
            )
        return str(model_path)

    if args.no_auto_find_embedding_model:
        return str(args.teacher_model_name)

    candidates = find_embedding_model_dirs(roots)
    if len(candidates) == 1:
        print(f"[AUTO] SecureBERT phát hiện tự động: {candidates[0]}", flush=True)
        return str(candidates[0])
    if len(candidates) > 1:
        listed = "\n".join(f"  - {p}" for p in candidates)
        raise RuntimeError(
            "Tìm thấy nhiều thư mục model HuggingFace. "
            "Dùng --embedding-model-path để chỉ định:\n" + listed
        )

    print(
        f"[WARN] Không tìm thấy SecureBERT local → sẽ tải từ HuggingFace: {args.teacher_model_name}",
        flush=True,
    )
    return str(args.teacher_model_name)


def require_payload_files(input_root: Path, payload_dir: Path) -> None:
    """Tìm và copy payload_256.npy + metadata.csv từ input_root. Báo lỗi nếu không có."""
    payload_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("payload_256.npy", "metadata.csv"):
        dst = payload_dir / fname
        if dst.exists():
            print(f"[OK] {fname} đã có sẵn: {dst}", flush=True)
            continue
        src = find_input_file(input_root, fname)
        if src is None:
            raise FileNotFoundError(
                f"\n[ERROR] Không tìm thấy '{fname}' trong {input_root}.\n"
                "Hãy upload kết quả extract_payload_dataset làm Kaggle dataset input."
            )
        print(f"[COPY] {fname}: {src} → {dst}", flush=True)
        shutil.copy2(src, dst)
        size_gb = dst.stat().st_size / 1024 ** 3
        print(f"[OK]   {fname} đã copy ({size_gb:.2f} GB)", flush=True)


# ── argument parser ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Kaggle retrain: load pre-extracted payload + offline SecureBERT, "
            "retrain student 1D-CNN từ đầu, build three-tier graph NPZ."
        )
    )
    parser.add_argument("--input-root", default="/kaggle/input",
                        help="Thư mục gốc Kaggle input (chứa tất cả dataset đã upload).")
    parser.add_argument("--work-dir", default="/kaggle/working/nt114_npz_work")
    parser.add_argument("--output-zip", default="/kaggle/working/graph_npz_artifact.zip")
    parser.add_argument("--reset-work-dir", action="store_true",
                        help="Xoá và tạo lại work-dir từ đầu.")

    # Teacher
    parser.add_argument("--teacher-model-name", default="ehsanaghaei/SecureBERT",
                        help="Fallback HuggingFace ID khi không tìm thấy model local.")
    parser.add_argument("--embedding-model-path", default="",
                        help="Đường dẫn thư mục SecureBERT local. Ghi đè auto-detect.")
    parser.add_argument("--no-auto-find-embedding-model", action="store_true",
                        help="Tắt tự động tìm model trong input/work roots.")
    parser.add_argument("--teacher-batch-size", type=int, default=128,
                        help="Batch size cho SecureBERT inference (128 phù hợp Kaggle T4/P100).")
    parser.add_argument("--teacher-max-length", type=int, default=512)

    # Student
    parser.add_argument("--student-epochs", type=int, default=30)
    parser.add_argument("--student-batch-size", type=int, default=256)
    parser.add_argument("--student-num-workers", type=int, default=2)
    parser.add_argument("--student-emb-batch-size", type=int, default=1024)

    # MITRE
    parser.add_argument("--mitre-batch-size", type=int, default=64)
    parser.add_argument(
        "--mitre-stix-url",
        default="https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json",
        help="URL tải enterprise-attack.json (chỉ dùng khi không có file local — cần internet).",
    )

    # Device
    parser.add_argument("--device", default="auto")

    # Graph
    parser.add_argument("--flow-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-packets-per-flow", type=int, default=20)
    parser.add_argument("--similarity-threshold", type=float, default=0.82)
    parser.add_argument("--packet-top-k", type=int, default=5)
    parser.add_argument("--flow-top-k", type=int, default=5)
    parser.add_argument("--graph-npz-name", default="graph_artifact_3tier_t082_k5.npz")
    parser.add_argument("--graph-meta-name", default="graph_artifact_3tier_t082_k5.meta.json")
    parser.add_argument("--sim-batch-size", type=int, default=50_000,
                        help="Packets/batch cho cosine similarity trên GPU.")

    return parser.parse_args()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    work_dir = Path(args.work_dir)
    output_zip = Path(args.output_zip)

    if args.reset_work_dir and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Payload files ──────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("[STEP 1/7] Payload dataset", flush=True)
    print("=" * 60, flush=True)

    payload_dir = work_dir / "data" / "interim" / "payload_dataset"
    payload_npy = payload_dir / "payload_256.npy"
    metadata_csv = payload_dir / "metadata.csv"

    require_payload_files(input_root, payload_dir)
    print(f"[OK] payload_256.npy shape: {_npy_shape(payload_npy)}", flush=True)

    # ── 2. Locate SecureBERT ──────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("[STEP 2/7] Locate SecureBERT model", flush=True)
    print("=" * 60, flush=True)

    embedding_model = resolve_embedding_model(args, roots=[input_root, work_dir])
    print(f"[OK] Embedding model: {embedding_model}", flush=True)

    # ── 3. Teacher targets ────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("[STEP 3/7] Teacher targets (SecureBERT inference)", flush=True)
    print("=" * 60, flush=True)

    processed_dir = work_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    teacher_npy = processed_dir / "teacher_targets.npy"

    # Cho phép tái sử dụng teacher_targets.npy đã có trong input (tiết kiệm GPU)
    if not teacher_npy.exists():
        if copy_input_artifact(input_root, "teacher_targets.npy", teacher_npy):
            meta_src = find_input_file(input_root, "teacher_targets.meta.json")
            if meta_src:
                shutil.copy2(meta_src, teacher_npy.with_suffix(".meta.json"))
            print("[SKIP] Dùng teacher_targets.npy từ input.", flush=True)

    if not teacher_npy.exists():
        run([
            sys.executable, "-u", "-m",
            "graphslm_ids.offline_path.preprocessing.build_teacher_targets",
            "--payload-npy", str(payload_npy),
            "--metadata-csv", str(metadata_csv),
            "--output-path", str(teacher_npy),
            "--model-name", embedding_model,
            "--batch-size", str(args.teacher_batch_size),
            "--max-length", str(args.teacher_max_length),
            "--device", args.device,
        ])
    else:
        print(f"[SKIP] teacher_targets.npy đã tồn tại: {teacher_npy}", flush=True)

    # ── 4. Train student CNN (luôn retrain từ đầu) ────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("[STEP 4/7] Train student 1D-CNN (retrain từ đầu)", flush=True)
    print("=" * 60, flush=True)

    student_dir = work_dir / "outputs" / "student_cnn"
    student_ckpt = student_dir / "student_cnn_best.pt"

    if student_ckpt.exists():
        print(f"[RETRAIN] Xoá checkpoint cũ: {student_ckpt}", flush=True)
        student_ckpt.unlink()

    run([
        sys.executable, "-u", "-m",
        "graphslm_ids.offline_path.training.train_student_cnn",
        "--payload-npy", str(payload_npy),
        "--teacher-npy", str(teacher_npy),
        "--output-dir", str(student_dir),
        "--batch-size", str(args.student_batch_size),
        "--epochs", str(args.student_epochs),
        "--num-workers", str(args.student_num_workers),
        "--device", args.device,
    ])

    # ── 5. Export student embeddings ──────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("[STEP 5/7] Export student embeddings", flush=True)
    print("=" * 60, flush=True)

    student_emb_npy = processed_dir / "student_embeddings.npy"
    if student_emb_npy.exists():
        student_emb_npy.unlink()

    run([
        sys.executable, "-u", "-m",
        "graphslm_ids.offline_path.training.export_student_embeddings",
        "--payload-npy", str(payload_npy),
        "--checkpoint", str(student_ckpt),
        "--output-path", str(student_emb_npy),
        "--batch-size", str(args.student_emb_batch_size),
        "--device", args.device,
    ])

    # ── 6. MITRE knowledge base ───────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("[STEP 6/7] MITRE knowledge base", flush=True)
    print("=" * 60, flush=True)

    mitre_dir = work_dir / "data" / "mitre"
    mitre_dir.mkdir(parents=True, exist_ok=True)
    mitre_stix = mitre_dir / "enterprise-attack.json"
    mitre_tech = mitre_dir / "mitre_techniques.csv"
    mitre_tac  = mitre_dir / "mitre_tactics.csv"
    mitre_edge = mitre_dir / "mitre_technique_tactic_edges.csv"
    mitre_emb  = mitre_dir / "mitre_techniques_embeddings.npy"

    for name, dst in [
        ("enterprise-attack.json",              mitre_stix),
        ("mitre_techniques.csv",                mitre_tech),
        ("mitre_tactics.csv",                   mitre_tac),
        ("mitre_technique_tactic_edges.csv",    mitre_edge),
        ("mitre_techniques_embeddings.npy",     mitre_emb),
    ]:
        copy_input_artifact(input_root, name, dst)

    if not (mitre_tech.exists() and mitre_edge.exists()):
        if not mitre_stix.exists():
            print(
                "[DOWNLOAD] enterprise-attack.json không có trong input → tải từ GitHub (cần internet).",
                flush=True,
            )
            mitre_stix.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(args.mitre_stix_url, mitre_stix)

        run([
            sys.executable, "-u", "-m",
            "graphslm_ids.offline_path.preprocessing.prepare_mitre_knowledge_base",
            "--input-json", str(mitre_stix),
            "--techniques-csv", str(mitre_tech),
            "--tactics-csv", str(mitre_tac),
            "--technique-tactic-edges-csv", str(mitre_edge),
            "--stats-json", str(mitre_dir / "mitre_export_stats.json"),
        ])
    else:
        print("[SKIP] MITRE CSVs đã có sẵn.", flush=True)

    if not mitre_emb.exists():
        run([
            sys.executable, "-u", "-m",
            "graphslm_ids.offline_path.preprocessing.build_mitre_technique_embeddings",
            "--techniques-csv", str(mitre_tech),
            "--output-path", str(mitre_emb),
            "--model-name", embedding_model,
            "--teacher-meta-json", str(teacher_npy.with_suffix(".meta.json")),
            "--batch-size", str(args.mitre_batch_size),
            "--device", args.device,
        ])
    else:
        print("[SKIP] mitre_techniques_embeddings.npy đã có sẵn.", flush=True)

    # ── 7. Build three-tier graph artifact ────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("[STEP 7/7] Build three-tier graph artifact", flush=True)
    print("=" * 60, flush=True)

    graph_npz = processed_dir / args.graph_npz_name
    graph_meta_json = processed_dir / args.graph_meta_name

    run([
        sys.executable, "-u", "-m",
        "graphslm_ids.offline_path.preprocessing.build_three_tier_graph_artifact",
        "--metadata-csv",                  str(metadata_csv),
        "--payload-npy",                   str(payload_npy),
        "--student-embedding-npy",         str(student_emb_npy),
        "--mitre-techniques-csv",          str(mitre_tech),
        "--mitre-technique-embeddings-npy", str(mitre_emb),
        "--mitre-technique-tactic-edges-csv", str(mitre_edge),
        "--output-npz",                    str(graph_npz),
        "--output-meta-json",              str(graph_meta_json),
        "--flow-timeout-seconds",          str(args.flow_timeout_seconds),
        "--max-packets-per-flow",          str(args.max_packets_per_flow),
        "--similarity-threshold",          str(args.similarity_threshold),
        "--packet-top-k",                  str(args.packet_top_k),
        "--flow-top-k",                    str(args.flow_top_k),
        "--device",                        args.device,
        "--sim-batch-size",                str(args.sim_batch_size),
    ])

    # ── Package output zip ────────────────────────────────────────────────────
    print("\n[PACKAGE] Đóng gói output...", flush=True)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        zf.write(graph_npz, arcname=graph_npz.name)
        zf.write(graph_meta_json, arcname=graph_meta_json.name)
        stats_path = payload_dir / "stats.json"
        if stats_path.exists():
            zf.write(stats_path, arcname="payload_stats.json")

    print("\n" + "=" * 60, flush=True)
    print("[DONE]", flush=True)
    print(f"  Graph NPZ : {graph_npz}  ({graph_npz.stat().st_size / 1024**3:.3f} GB)", flush=True)
    print(f"  Meta JSON : {graph_meta_json}", flush=True)
    print(f"  Zip output: {output_zip}  ({output_zip.stat().st_size / 1024**3:.3f} GB)", flush=True)
    print("=" * 60, flush=True)

    try:
        with graph_meta_json.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        keys = [
            "num_packets", "num_flows", "num_techniques", "num_tactics",
            "num_packet_technique_edges", "num_flow_technique_edges",
        ]
        print(json.dumps({k: meta.get(k) for k in keys}, indent=2), flush=True)
    except Exception as exc:
        print(f"[WARN] Không đọc được graph meta: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
