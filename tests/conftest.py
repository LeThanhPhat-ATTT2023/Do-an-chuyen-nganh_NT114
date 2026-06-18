from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# Make project-root packages (e.g. ``scripts.eval``) importable in tests.
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
