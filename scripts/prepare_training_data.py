from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from ai_env.train import prepare_data


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare repository fine-tuning data from source files."
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "training_data"),
        help="Directory to write generated training files.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Maximum characters per chunk.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of chunks to reserve for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset shuffling.",
    )
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    prepare_data(
        output_dir=Path(args.output_dir),
        max_chars=args.max_chars,
        val_split=args.validation_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
