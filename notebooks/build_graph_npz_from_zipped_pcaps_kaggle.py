from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n$", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd or REPO_DIR), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build only the three-tier graph NPZ from zipped/raw PCAP inputs on Kaggle."
    )
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--work-dir", default="/kaggle/working/nt114_npz_work")
    parser.add_argument("--output-zip", default="/kaggle/working/graph_npz_artifact.zip")
    parser.add_argument("--reset-work-dir", action="store_true")

    parser.add_argument("--payload-length", type=int, default=256)
    parser.add_argument("--max-packets-per-file", type=int, default=None)
    parser.add_argument("--include-empty-payload", action="store_true")
    parser.add_argument("--payload-extract-workers", type=int, default=0)
    parser.add_argument("--log-every-packets", type=int, default=100_000)
    parser.add_argument("--write-batch-size", type=int, default=100_000)

    parser.add_argument("--teacher-model-name", default="ehsanaghaei/SecureBERT")
    parser.add_argument("--teacher-batch-size", type=int, default=32)
    parser.add_argument("--teacher-max-length", type=int, default=512)
    parser.add_argument("--student-epochs", type=int, default=30)
    parser.add_argument("--student-batch-size", type=int, default=256)
    parser.add_argument("--student-num-workers", type=int, default=2)
    parser.add_argument("--student-emb-batch-size", type=int, default=1024)
    parser.add_argument("--mitre-batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--flow-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-packets-per-flow", type=int, default=20)
    parser.add_argument("--similarity-threshold", type=float, default=0.82)
    parser.add_argument("--packet-top-k", type=int, default=5)
    parser.add_argument("--flow-top-k", type=int, default=5)
    parser.add_argument("--graph-npz-name", default="graph_artifact_3tier_t082_k5.npz")
    parser.add_argument("--graph-meta-name", default="graph_artifact_3tier_t082_k5.meta.json")

    parser.add_argument(
        "--mitre-stix-url",
        default="https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json",
    )
    return parser.parse_args()


def find_input_file(input_root: Path, name: str) -> Path | None:
    matches = sorted(input_root.glob(f"**/{name}"))
    return matches[0] if matches else None


def archive_paths(input_root: Path) -> list[Path]:
    patterns = [
        "**/*.zip",
        "**/*.rar",
        "**/*.7z",
        "**/*.tar",
        "**/*.tar.gz",
        "**/*.tgz",
    ]
    found: dict[str, Path] = {}
    for pattern in patterns:
        for item in input_root.glob(pattern):
            if item.is_file():
                found[str(item.resolve())] = item
    return sorted(found.values(), key=lambda p: str(p))


def unpack_archive(path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    suffix = "".join(path.suffixes).lower()
    print(f"Unpack: {path} -> {dest}", flush=True)
    if suffix.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dest)
        return

    seven_zip = shutil.which("7z") or shutil.which("7za")
    if suffix.endswith((".rar", ".7z")) and seven_zip:
        run([seven_zip, "x", str(path), f"-o{dest}", "-y"])
        return

    shutil.unpack_archive(str(path), str(dest))


def discover_pcaps(input_root: Path, raw_dir: Path) -> list[Path]:
    patterns = [
        str(raw_dir / "**" / "*.pcap"),
        str(raw_dir / "**" / "*.pcapng"),
        str(input_root / "**" / "*.pcap"),
        str(input_root / "**" / "*.pcapng"),
    ]
    found: dict[str, Path] = {}
    for pattern in patterns:
        for item in glob.glob(pattern, recursive=True):
            path = Path(item)
            if path.is_file() and path.suffix.lower() in {".pcap", ".pcapng"}:
                found[str(path.resolve())] = path
    return sorted(found.values(), key=lambda p: str(p))


def copy_input_artifact(input_root: Path, name: str, dst: Path) -> bool:
    src = find_input_file(input_root, name)
    if src is None or dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def verify_extractor_support() -> None:
    help_text = subprocess.check_output(
        [sys.executable, "-m", "graphslm_ids.offline_path.preprocessing.extract_payload_dataset", "--help"],
        cwd=str(REPO_DIR),
        text=True,
    )
    if "--stream-to-disk" not in help_text or "--num-workers" not in help_text:
        raise RuntimeError("Repo revision too old: extractor must support --stream-to-disk and --num-workers.")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    work_dir = Path(args.work_dir)
    output_zip = Path(args.output_zip)
    raw_dir = work_dir / "raw_unpacked"

    if args.reset_work_dir and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    verify_extractor_support()

    archives = archive_paths(input_root)
    print(f"Archives found: {len(archives)}", flush=True)
    for idx, archive in enumerate(archives):
        unpack_archive(archive, raw_dir / f"archive_{idx:03d}_{archive.stem}")

    pcaps = discover_pcaps(input_root=input_root, raw_dir=raw_dir)
    if not pcaps:
        raise RuntimeError("No PCAP/PCAPNG found in input archives or raw Kaggle input.")

    manifest = work_dir / "pcap_manifest.txt"
    manifest.write_text("\n".join(str(p) for p in pcaps), encoding="utf-8")
    print(f"PCAP files: {len(pcaps)}", flush=True)
    for item in pcaps[:10]:
        print(f" - {item}", flush=True)
    if len(pcaps) > 10:
        print(f" ... {len(pcaps) - 10} more", flush=True)

    payload_dir = work_dir / "data" / "interim" / "payload_dataset"
    payload_npy = payload_dir / "payload_256.npy"
    metadata_csv = payload_dir / "metadata.csv"
    raw_pcap_globs = [
        str(raw_dir / "**" / "*.pcap"),
        str(raw_dir / "**" / "*.pcapng"),
        str(input_root / "**" / "*.pcap"),
        str(input_root / "**" / "*.pcapng"),
    ]

    if not (payload_npy.exists() and metadata_csv.exists()):
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "graphslm_ids.offline_path.preprocessing.extract_payload_dataset",
            "--input-glob",
            *raw_pcap_globs,
            "--output-dir",
            str(payload_dir),
            "--payload-length",
            str(args.payload_length),
            "--stream-to-disk",
            "--stream-mode",
            "single-pass",
            "--num-workers",
            str(args.payload_extract_workers),
            "--write-batch-size",
            str(args.write_batch_size),
            "--log-every-packets",
            str(args.log_every_packets),
        ]
        if args.max_packets_per_file is not None:
            cmd += ["--max-packets-per-file", str(args.max_packets_per_file)]
        if args.include_empty_payload:
            cmd += ["--include-empty-payload"]
        run(cmd)

    processed_dir = work_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    teacher_npy = processed_dir / "teacher_targets.npy"
    student_emb_npy = processed_dir / "student_embeddings.npy"
    student_dir = work_dir / "outputs" / "student_cnn"
    student_ckpt = student_dir / "student_cnn_best.pt"

    if copy_input_artifact(input_root, "teacher_targets.npy", teacher_npy):
        src_meta = find_input_file(input_root, "teacher_targets.meta.json")
        if src_meta:
            shutil.copy2(src_meta, teacher_npy.with_suffix(".meta.json"))
    copy_input_artifact(input_root, "student_cnn_best.pt", student_ckpt)
    copy_input_artifact(input_root, "student_embeddings.npy", student_emb_npy)

    if not teacher_npy.exists() and not student_ckpt.exists() and not student_emb_npy.exists():
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "graphslm_ids.offline_path.preprocessing.build_teacher_targets",
                "--payload-npy",
                str(payload_npy),
                "--metadata-csv",
                str(metadata_csv),
                "--output-path",
                str(teacher_npy),
                "--model-name",
                args.teacher_model_name,
                "--batch-size",
                str(args.teacher_batch_size),
                "--max-length",
                str(args.teacher_max_length),
                "--device",
                args.device,
            ]
        )

    if not student_ckpt.exists() and not student_emb_npy.exists():
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "graphslm_ids.offline_path.training.train_student_cnn",
                "--payload-npy",
                str(payload_npy),
                "--teacher-npy",
                str(teacher_npy),
                "--output-dir",
                str(student_dir),
                "--batch-size",
                str(args.student_batch_size),
                "--epochs",
                str(args.student_epochs),
                "--num-workers",
                str(args.student_num_workers),
                "--device",
                args.device,
            ]
        )

    if not student_emb_npy.exists():
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "graphslm_ids.offline_path.training.export_student_embeddings",
                "--payload-npy",
                str(payload_npy),
                "--checkpoint",
                str(student_ckpt),
                "--output-path",
                str(student_emb_npy),
                "--batch-size",
                str(args.student_emb_batch_size),
                "--device",
                args.device,
            ]
        )

    mitre_dir = work_dir / "data" / "mitre"
    mitre_dir.mkdir(parents=True, exist_ok=True)
    mitre_stix = mitre_dir / "enterprise-attack.json"
    mitre_tech = mitre_dir / "mitre_techniques.csv"
    mitre_tac = mitre_dir / "mitre_tactics.csv"
    mitre_edge = mitre_dir / "mitre_technique_tactic_edges.csv"
    mitre_emb = mitre_dir / "mitre_techniques_embeddings.npy"

    for name, dst in [
        ("mitre_techniques.csv", mitre_tech),
        ("mitre_tactics.csv", mitre_tac),
        ("mitre_technique_tactic_edges.csv", mitre_edge),
        ("mitre_techniques_embeddings.npy", mitre_emb),
        ("enterprise-attack.json", mitre_stix),
    ]:
        copy_input_artifact(input_root, name, dst)

    if not (mitre_tech.exists() and mitre_edge.exists()):
        if not mitre_stix.exists():
            print(f"Download MITRE STIX: {args.mitre_stix_url}", flush=True)
            urllib.request.urlretrieve(args.mitre_stix_url, mitre_stix)
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "graphslm_ids.offline_path.preprocessing.prepare_mitre_knowledge_base",
                "--input-json",
                str(mitre_stix),
                "--techniques-csv",
                str(mitre_tech),
                "--tactics-csv",
                str(mitre_tac),
                "--technique-tactic-edges-csv",
                str(mitre_edge),
                "--stats-json",
                str(mitre_dir / "mitre_export_stats.json"),
            ]
        )

    if not mitre_emb.exists():
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "graphslm_ids.offline_path.preprocessing.build_mitre_technique_embeddings",
                "--techniques-csv",
                str(mitre_tech),
                "--output-path",
                str(mitre_emb),
                "--teacher-meta-json",
                str(teacher_npy.with_suffix(".meta.json")),
                "--batch-size",
                str(args.mitre_batch_size),
                "--device",
                args.device,
            ]
        )

    graph_npz = processed_dir / args.graph_npz_name
    graph_meta_json = processed_dir / args.graph_meta_name
    if not (graph_npz.exists() and graph_meta_json.exists()):
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "graphslm_ids.offline_path.preprocessing.build_three_tier_graph_artifact",
                "--metadata-csv",
                str(metadata_csv),
                "--payload-npy",
                str(payload_npy),
                "--student-embedding-npy",
                str(student_emb_npy),
                "--mitre-techniques-csv",
                str(mitre_tech),
                "--mitre-technique-embeddings-npy",
                str(mitre_emb),
                "--mitre-technique-tactic-edges-csv",
                str(mitre_edge),
                "--output-npz",
                str(graph_npz),
                "--output-meta-json",
                str(graph_meta_json),
                "--flow-timeout-seconds",
                str(args.flow_timeout_seconds),
                "--max-packets-per-flow",
                str(args.max_packets_per_flow),
                "--similarity-threshold",
                str(args.similarity_threshold),
                "--packet-top-k",
                str(args.packet_top_k),
                "--flow-top-k",
                str(args.flow_top_k),
            ]
        )

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        zf.write(graph_npz, arcname=graph_npz.name)
        zf.write(graph_meta_json, arcname=graph_meta_json.name)
        stats_path = payload_dir / "stats.json"
        if stats_path.exists():
            zf.write(stats_path, arcname="payload_stats.json")
        if manifest.exists():
            zf.write(manifest, arcname="pcap_manifest.txt")

    print("\n[OK] Graph NPZ:", graph_npz, flush=True)
    print("[OK] Graph meta:", graph_meta_json, flush=True)
    print("[OK] Download zip:", output_zip, flush=True)
    print("[OK] Graph NPZ GB:", round(graph_npz.stat().st_size / 1024**3, 3), flush=True)
    print("[OK] Zip GB:", round(output_zip.stat().st_size / 1024**3, 3), flush=True)

    try:
        with graph_meta_json.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        keys = [
            "num_packets",
            "num_flows",
            "num_techniques",
            "num_tactics",
            "num_packet_technique_edges",
            "num_flow_technique_edges",
        ]
        print(json.dumps({key: meta.get(key) for key in keys}, indent=2), flush=True)
    except Exception as exc:
        print("Could not print graph meta summary:", repr(exc), flush=True)


if __name__ == "__main__":
    main()
