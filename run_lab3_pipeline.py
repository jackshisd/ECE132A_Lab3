from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_step(script: Path) -> None:
    cmd = [sys.executable, str(script)]
    print(f"Running {script.name} ...")
    subprocess.run(cmd, check=True)


def main() -> None:
    base = Path(__file__).resolve().parent
    order = [
        "lab31.py",
        "sdr_lab3.py",
        "lab33.py",
        "lab34.py",
        "lab35.py",
    ]

    for name in order:
        run_step(base / name)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
