from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from ai_env.web import prepare_web_data


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare web snapshot fine-tuning data from public URLs."
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        required=True,
        help="Public URLs to snapshot into training data.",
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
        "--val-split",
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
    prepare_web_data(
        output_dir=Path(args.output_dir),
        urls=args.urls,
        max_chars=args.max_chars,
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
