#!/usr/bin/env python3
"""Run dataset preparation and all training / evaluation stages."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEPS = [
    ROOT / "scripts" / "download_and_prepare.py",
    ROOT / "training" / "train_structured.py",
    ROOT / "training" / "train_lstm.py",
    ROOT / "training" / "train_vision.py",
    ROOT / "training" / "train_fusion.py",
    ROOT / "evaluation" / "evaluate.py",
]


def main() -> None:
    for script in STEPS:
        print(f"\n=== {script.name} ===", flush=True)
        completed = subprocess.run([sys.executable, str(script)], cwd=ROOT)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
