"""Public training entry point for the three released Gaussolve backbones."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


BACKBONES = {
    "3dgs": "train_gaussolve.py",
    "mcmc": "train_gaussolve_mcmc.py",
    "reduced": "train_gaussolve_reduced.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backbone", required=True, choices=tuple(BACKBONES))
    known, remaining = parser.parse_known_args()
    entry = Path(__file__).resolve().parent / BACKBONES[known.backbone]
    sys.argv = [str(entry), *remaining]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
