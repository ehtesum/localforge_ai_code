from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from .config import ROOT
from .workspace import (
    chunk_text,
    is_text_file,
    read_text_file,
    should_ignore,
)
from .data import (
    iter_repo_chunks as _iter_repo_chunks,
    prepare_repo_dataset,
    read_jsonl,
    write_jsonl,
)


def matches_category(path: Path, categories: set[str]) -> bool:
    from .data import _matches_category
    return _matches_category(path.as_posix(), categories)


def iter_repo_chunks(
    root: Path,
    max_chars: int = 2000,
    categories: set[str] | None = None,
) -> Iterable[dict]:
    for chunk in _iter_repo_chunks(root, max_chars=max_chars, categories=categories):
        yield chunk.to_dict()


def prepare_data(
    output_dir: Path = ROOT / "training_data",
    max_chars: int = 2000,
    val_split: float = 0.1,
    seed: int = 42,
    categories: set[str] | None = None,
) -> None:
    result = prepare_repo_dataset(
        output_dir=output_dir,
        root=ROOT,
        max_chars=max_chars,
        val_split=val_split,
        seed=seed,
        categories=categories,
    )

    training_records = list(read_jsonl(result.train_path))
    validation_records = list(read_jsonl(result.validation_path))
    source_files = {r["path"] for r in training_records + validation_records}

    print(f"Prepared training data from {ROOT}")
    print(f" - source files: {len(source_files)}")
    print(f" - total chunks: {len(training_records) + len(validation_records)}")
    print(f" - training examples: {len(training_records)}")
    print(f" - validation examples: {len(validation_records)}")
    print(f"Data written to: {result.train_path} and {result.validation_path}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repository fine-tuning dataset preparation.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-data",
        help="Collect repository files and create JSONL chunks for fine-tuning.",
    )
    prepare_parser.add_argument(
        "--output-dir",
        default=str(ROOT / "training_data"),
        help="Directory to write generated training files.",
    )
    prepare_parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Maximum characters per chunk.",
    )
    prepare_parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of chunks to reserve for validation.",
    )
    prepare_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset shuffling.",
    )
    prepare_parser.add_argument(
        "--categories",
        nargs="+",
        choices=["all", "python", "devops", "frontend"],
        default=["all"],
        help="Topic categories to include when preparing training data.",
    )

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    if args.subcommand == "prepare-data":
        prepare_data(
            output_dir=Path(args.output_dir),
            max_chars=args.max_chars,
            val_split=args.val_split,
            seed=args.seed,
            categories=set(args.categories),
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
