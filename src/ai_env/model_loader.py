"""Shared model loading utilities.

Centralises checkpoint resolution, tokenizer loading, and model construction
so that ``agent.py`` and ``inference.py`` (and any future consumer) share a
single, well-tested implementation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import torch

try:  # transformers is an optional dependency at runtime
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "The 'transformers' package is required for model loading. "
        "Install it with: pip install -r requirements.txt"
    ) from exc


@dataclass
class LoadResult:
    """Container for a successfully loaded model + tokenizer."""

    model: "AutoModelForCausalLM"
    tokenizer: "AutoTokenizer"
    device: str
    model_path: Path
    used_checkpoint: Optional[str] = None
    extra: dict = field(default_factory=dict)


def detect_device(prefer_cuda: bool = True) -> str:
    """Return the best available torch device."""
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_checkpoint(checkpoint_dir: Path | str, fallback_remote: Optional[str] = None) -> Tuple[Optional[Path], Optional[str]]:
    """Find the latest ``checkpoint-N`` subdirectory in ``checkpoint_dir``.

    Returns ``(model_path, checkpoint_name)``. ``model_path`` is ``None`` if no
    checkpoint exists and no ``fallback_remote`` is provided.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return (Path(fallback_remote) if fallback_remote else None), None

    subdirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    if subdirs:
        latest = sorted(subdirs, key=lambda x: int(x.split("-")[1]))[-1]
        return checkpoint_dir / latest, latest

    # No numbered subdirs — use the directory itself if non-empty
    if any(checkpoint_dir.iterdir()):
        return checkpoint_dir, None

    return (Path(fallback_remote) if fallback_remote else None), None


def _load_tokenizer(path: Path | str) -> "AutoTokenizer":
    try:
        return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    except Exception as primary_error:
        try:
            return AutoTokenizer.from_pretrained(
                str(path), trust_remote_code=True, use_fast=False
            )
        except Exception:
            # Re-raise the original, more informative error
            raise primary_error


def load_model_and_tokenizer(
    checkpoint_dir: Path | str = "./checkpoints",
    model_name: str = "Qwen/Qwen2.5-Coder-7B",
    device: Optional[str] = None,
) -> LoadResult:
    """Load the fine-tuned model and tokenizer, resolving the latest checkpoint.

    Behaviour:
    1. If ``checkpoint_dir`` contains a ``checkpoint-N`` subdir, use the latest.
    2. Else, use ``checkpoint_dir`` itself if it is non-empty.
    3. Else, fall back to the remote ``model_name`` (will be downloaded).
    """
    if device is None:
        device = detect_device()

    model_path, used_checkpoint = resolve_checkpoint(checkpoint_dir, fallback_remote=model_name)
    if model_path is None:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir} and no fallback model provided."
        )

    print(f"[MODEL] Loading from {model_path}" + (f" (checkpoint: {used_checkpoint})" if used_checkpoint else ""))

    tokenizer = _load_tokenizer(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    return LoadResult(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_path=Path(model_path),
        used_checkpoint=used_checkpoint,
    )


def generate_text(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    device: Optional[str] = None,
    top_p: float = 0.9,
) -> str:
    """Generate text from a prompt and return the decoded continuation.

    The returned string is the portion *after* the prompt.
    """
    device = device or next(model.parameters()).device.type
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full = tokenizer.decode(output[0], skip_special_tokens=True)
    # Trim the prompt portion
    if full.startswith(prompt):
        return full[len(prompt):].strip()
    return full.strip()
