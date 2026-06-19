"""Replay a PCAP through the online IDS runtime and PRINT the full execution flow.

This is a demo/observability wrapper around the production entrypoint
``graphslm_ids.runtime.pipeline.run_runtime_pipeline``. The production CLI runs the
slow path on a background thread and silently persists the XAI report to the cold
store (``outputs/runtime/cold_store.jsonl``, record ``type: "report"``). For a live
demo you usually want to *watch* the alert + grounded report appear inline, in order,
as packets flow through the system.

So this script:
  1. Reads packets with the SAME dpkt decoder the offline extractor uses
     (reused from ``run_runtime_pipeline.iter_packets`` -> offline parity).
  2. Feeds up to ``--max-packets`` packets one-by-one through ``pipeline.on_packet``
     (the fast path: payload -> flow -> MSEE technique evidence -> hot graph ->
      subgraph -> HGT -> policy -> alert dispatch).
  3. Does NOT start the background slow-path worker. Instead, every time the policy
     raises an alert, it drains the slow queue and runs ``process_job`` SYNCHRONOUSLY,
     printing the evidence-grounded XAI report right there in the flow.

Nothing about the model or the pipeline changes — only *when* and *where* the output
is shown. Reports are still persisted to the cold store exactly as in production.

Example
-------
    D:\\v\\nt114\\Scripts\\python.exe scripts/demo/replay_pcap_demo.py ^
        --config configs/pipeline.example.yaml ^
        --input data/raw/SqlInjection/<capture>.pcap ^
        --max-packets 600

Add ``--trace`` to print a one-line decision for EVERY packet (verbose, ~600 lines).
The SLM narrative (Ollama) is optional: if the endpoint is unreachable the slow path
falls back to a deterministic, evidence-grounded template report (tier 3) and the demo
still works end-to-end on a bare server.
"""

from __future__ import annotations

import argparse
import queue
import sys
from pathlib import Path

# Make ``src/`` importable when run directly (python scripts/demo/replay_pcap_demo.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from graphslm_ids.runtime import FastPathPipeline, PipelineConfig  # noqa: E402
from graphslm_ids.runtime.pipeline.run_runtime_pipeline import iter_packets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a PCAP through the online IDS runtime and print the "
        "fast-path decision + slow-path XAI report for every alert."
    )
    parser.add_argument("--config", default="configs/pipeline.example.yaml")
    parser.add_argument("--input", required=True, help="PCAP/PCAPNG file to replay.")
    parser.add_argument("--max-packets", type=int, default=600)
    parser.add_argument(
        "--device",
        default=None,
        help="Override hgt.device from the config (e.g. cpu / cuda).",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print a one-line fast-path decision for EVERY packet (verbose).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print a progress line every N packets (0 disables).",
    )
    return parser.parse_args()


def _hr(char: str = "=", width: int = 78) -> str:
    return char * width


def _print_flow_summary(snapshot: dict) -> None:
    """Print the 5-tuple + packet count for the alerting flow, if available."""
    flow = (snapshot or {}).get("flow", {}) or {}
    packets = (snapshot or {}).get("packets", []) or []
    src = flow.get("src_ip", "?")
    dst = flow.get("dst_ip", "?")
    sport = flow.get("src_port", "?")
    dport = flow.get("dst_port", "?")
    proto = flow.get("protocol", "?")
    print(f"    flow 5-tuple : {src}:{sport} -> {dst}:{dport} / {proto}")
    print(f"    packets seen : {len(packets)}")


def main() -> int:
    args = parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    if args.device:
        cfg.hgt.device = args.device

    print(_hr())
    print("GraphSLM-IDS — online runtime replay (end-to-end execution flow)")
    print(_hr())
    print(f"config      : {args.config}")
    print(f"pcap input  : {args.input}")
    print(f"max packets : {args.max_packets}")
    print(f"hgt ckpt    : {cfg.hgt.checkpoint}  (device={cfg.hgt.device})")
    print(f"graph meta  : {cfg.hgt.graph_meta_json}")
    print(f"alert thr   : {cfg.policy.alert_threshold}  benign={cfg.policy.benign_labels}")
    print(f"SLM backend : {cfg.slm.backend} @ {cfg.slm.endpoint} ({cfg.slm.model_name})")
    print(_hr())
    print(
        "Fast path : payload -> flow -> MSEE(PMI+procedure) technique evidence ->\n"
        "            hot graph -> k-hop subgraph -> HGT infer -> policy -> alert\n"
        "Slow path : hydrate context -> evidence bundle (+counterfactual) -> rank ->\n"
        "            serialize subgraph -> SLM (VG2R verify/repair) -> grounded report\n"
        "(slow path runs SYNCHRONOUSLY here so the report prints inline per alert)"
    )
    print(_hr())

    # Build the real pipeline, but DO NOT start the background slow worker — we drain
    # the queue ourselves so each report is printed in order, right after its packet.
    pipeline = FastPathPipeline(cfg)
    # process_job needs the hot buffer adapter only when the graph store is disabled
    # (mirrors FastPathPipeline.start_slow_worker).
    slow_hot_buffer = None if pipeline.graph_store is not None else pipeline.hot_source

    count = 0
    alerts = 0
    reports = 0
    seen_alert_flows: set[str] = set()

    for packet in iter_packets(args.input):
        if count >= args.max_packets:
            break
        result = pipeline.on_packet(packet)
        count += 1

        if args.trace:
            mark = "ALERT" if result.alert_id else "ok"
            print(
                f"  [{count:>4}] flow={result.flow_id} "
                f"label={result.label:<18} conf={result.confidence:.4f} [{mark}]"
            )
        elif args.progress_every and count % args.progress_every == 0:
            print(f"  ... processed {count} packets, alerts so far: {alerts}")

        if result.alert_id is None:
            continue

        alerts += 1
        seen_alert_flows.add(result.flow_id)

        # Drain every job the dispatcher enqueued for this packet and run the slow
        # path synchronously so the grounded report appears inline, in flow order.
        while True:
            try:
                job = pipeline.slow_queue.get_nowait()
            except queue.Empty:
                break
            if job is None:
                continue
            slow_result = pipeline.slow_worker.process_job(
                job,
                hot_buffer=slow_hot_buffer,
                cold_store=pipeline.cold_store,
            )
            reports += 1

            print()
            print(_hr("#"))
            print(f"# ALERT {job.alert_id}  (packet #{count})")
            print(_hr("#"))
            print(f"    predicted    : {job.predicted_label}  (confidence {job.confidence:.4f})")
            print(f"    threshold    : {job.alert_threshold}")
            _print_flow_summary(job.subgraph_snapshot)
            top = ", ".join(
                f"{c.get('label')}={c.get('prob', 0.0):.3f}" for c in (job.top_classes or [])[:5]
            )
            if top:
                print(f"    top classes  : {top}")
            tier = getattr(slow_result, "fallback_tier", "?")
            validation = getattr(slow_result, "validation", None)
            passed = getattr(validation, "overall_pass", None)
            print(f"    report tier  : {tier} (1=SLM, 2=SLM+repair, 3=template fallback)")
            if passed is not None:
                print(f"    verifier     : {'PASS' if passed else 'FAIL'}")
            print(_hr("-"))
            print("    XAI REPORT (evidence-grounded):")
            print(_hr("-"))
            report_text = getattr(slow_result, "report", "") or "(empty)"
            for line in report_text.splitlines():
                print(f"    {line}")
            print(_hr("#"))
            print()

    print()
    print(_hr())
    print("REPLAY SUMMARY")
    print(_hr())
    print(f"  packets processed : {count}")
    print(f"  alerts raised     : {alerts}")
    print(f"  reports generated : {reports}")
    print(f"  distinct flows alerted : {len(seen_alert_flows)}")
    if cfg.graph_store.enabled:
        print(f"  graph store       : {cfg.graph_store.root}")
    else:
        print(f"  cold store (jsonl): {cfg.cold_store_path}")
    print(_hr())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
