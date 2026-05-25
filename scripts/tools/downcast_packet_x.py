"""Downcast packet_x in an existing v3 graph npz from float32 to float16.

One-time tool so an artifact built before the float16 change does not need a
full ~2.5h rebuild. All non-packet arrays are copied verbatim.

Usage:
    python scripts/tools/downcast_packet_x.py IN.npz OUT.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def downcast_packet_x(src_npz: Path, out_npz: Path) -> None:
    src_npz = Path(src_npz)
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    with np.load(src_npz, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if "packet_x" in arrays:
        arrays["packet_x"] = np.asarray(arrays["packet_x"], dtype=np.float32).astype(
            np.float16
        )
    np.savez(str(out_npz), **arrays)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    downcast_packet_x(Path(argv[1]), Path(argv[2]))
    print(f"wrote {argv[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
