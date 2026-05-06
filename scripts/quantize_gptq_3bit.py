#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3-bit GPTQ quantization for TinyLlama-Chat using AutoGPTQ.

Designed for Windows + NVIDIA CUDA. Run from repo root:

    python scripts/quantize_gptq_3bit.py

Or with options:

    python scripts/quantize_gptq_3bit.py --output-dir models/gptq_3bit --calibration-samples 256
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


# ---------------------------------------------------------------------------
# Defaults aligned with this repository (TinyLlama Chat, WikiText-2 calibration)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_OUTPUT_REL = Path("models") / "gptq_3bit"
DEFAULT_CALIBRATION_SAMPLES = 256
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_MIN_TEXT_CHARS = 200

# File extensions counted as on-disk *weights* for size comparison (fair vs saved GPTQ).
_WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin"})


def _configure_logging(verbose: bool) -> None:
    """Structured console logging (no third-party log config)."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _repo_root() -> Path:
    """Directory containing `models/`, `scripts/`, etc. (parent of `scripts/`)."""
    return Path(__file__).resolve().parent.parent


def _require_cuda(logger: logging.Logger) -> torch.device:
    """
    Ensure a CUDA device is available.

    AutoGPTQ GPU quantization expects NVIDIA + CUDA. CPU fallback is not used here
    to avoid silent multi-hour runs or unsupported paths on Windows.
    """
    if not torch.cuda.is_available():
        logger.error(
            "CUDA is not available. Install a CUDA-enabled PyTorch build (see README) "
            'and verify with: python -c "import torch; print(torch.cuda.is_available())"'
        )
        raise SystemExit(2)
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    logger.info("Using GPU: %s (cuda:%s)", name, device.index)
    return device


def _log_vram(logger: logging.Logger, label: str) -> None:
    """Print driver-reported free/total and PyTorch allocator stats (Windows-friendly)."""
    if not torch.cuda.is_available():
        return
    free_b, total_b = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()
    logger.info(
        "%s | VRAM: free %.2f / total %.2f GiB | torch alloc %.2f | reserved %.2f | peak alloc %.2f GiB",
        label,
        free_b / (1024**3),
        total_b / (1024**3),
        alloc / (1024**3),
        reserved / (1024**3),
        peak / (1024**3),
    )


def _weights_disk_bytes(root: Path) -> int:
    """Total bytes of `.safetensors` / `.bin` files under `root` (sharded models supported)."""
    total = 0
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in _WEIGHT_SUFFIXES:
            total += path.stat().st_size
    return total


def _format_gib(n_bytes: int) -> str:
    return f"{n_bytes / (1024 ** 3):.3f} GiB"


def _resolve_original_snapshot_dir(model_id: str, logger: logging.Logger) -> Path:
    """
    Local directory containing the *baseline* HF weights for size comparison.

    - If `model_id` is a directory, use it directly.
    - Otherwise download / resolve the HF cache snapshot via `huggingface_hub`.
    """
    local = Path(model_id)
    if local.is_dir():
        logger.info("Using local model directory for baseline size: %s", local.resolve())
        return local.resolve()
    logger.info("Resolving Hugging Face snapshot for baseline size: %s", model_id)
    path = Path(
        snapshot_download(
            repo_id=model_id,
            repo_type="model",
        )
    )
    return path.resolve()


def _print_summary_banner(
    logger: logging.Logger,
    *,
    original_bytes: int,
    quantized_bytes: int,
    peak_quant_gib: float,
    post_save_alloc_gib: float,
    post_save_reserved_gib: float,
    post_save_driver_free_gib: float,
    post_save_driver_total_gib: float,
) -> None:
    """Single clean block for disk + GPU stats (easy to screenshot / log)."""
    ratio = (original_bytes / quantized_bytes) if quantized_bytes > 0 else float("nan")
    reduction = (1.0 - quantized_bytes / original_bytes) * 100 if original_bytes > 0 else float("nan")
    sep = "=" * 70
    lines = [
        sep,
        "SUMMARY",
        sep,
        f"  Disk (weight files: .safetensors, .bin)",
        f"    Original (HF snapshot):     {_format_gib(original_bytes):>12}  ({original_bytes:,} bytes)",
        f"    Quantized (saved output):   {_format_gib(quantized_bytes):>12}  ({quantized_bytes:,} bytes)",
        f"    Compression ratio:          {ratio:>12.2f}x  (smaller is better; quantized = original / ratio)",
        f"    Size reduction:             {reduction:>11.1f}%",
        sep,
        f"  GPU memory (PyTorch cuda:0)",
        f"    Peak during quantize():     {peak_quant_gib:>12.3f} GiB  (torch.cuda.max_memory_allocated)",
        f"    After save (allocated):     {post_save_alloc_gib:>12.3f} GiB",
        f"    After save (reserved):      {post_save_reserved_gib:>12.3f} GiB",
        f"    Driver heap free / total:   {post_save_driver_free_gib:>6.2f} / {post_save_driver_total_gib:>6.2f} GiB  (torch.cuda.mem_get_info)",
        sep,
    ]
    for line in lines:
        logger.info(line)


def _build_calibration_encodings(
    tokenizer: Any,
    *,
    num_samples: int,
    max_seq_len: int,
    min_chars: int,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """
    Build calibration examples for GPTQ.

    WikiText-2 raw train split is a standard, lightweight choice: no extra downloads
    beyond `datasets`, and snippets are long enough for stable Hessian estimates.
    """
    logger.info(
        "Loading WikiText-2 (wikitext-2-raw-v1 train) for up to %d calibration rows...",
        num_samples,
    )
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    except Exception as exc:
        logger.exception("Failed to load WikiText-2: %s", exc)
        raise

    texts: list[str] = []
    for row in tqdm(ds, desc="Selecting calibration texts", unit="row"):
        text = (row.get("text") or "").strip()
        if len(text) < min_chars:
            continue
        if text.startswith(" ="):
            continue
        texts.append(text)
        if len(texts) >= num_samples:
            break

    if len(texts) < num_samples:
        logger.warning(
            "Only collected %d calibration texts (requested %d). Continuing with fewer samples.",
            len(texts),
            num_samples,
        )

    examples: list[dict[str, Any]] = []
    for text in tqdm(texts, desc="Tokenizing for GPTQ", unit="ex"):
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=True,
        )
        examples.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
            }
        )

    logger.info("Prepared %d calibration examples (max_seq_len=%d).", len(examples), max_seq_len)
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize TinyLlama-1.1B-Chat with AutoGPTQ (3-bit, safetensors)."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id or local path (default: TinyLlama Chat v1.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: <repo>/{DEFAULT_OUTPUT_REL.as_posix()}).",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLES,
        help="Number of WikiText-2 snippets for calibration.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help="Max sequence length for each calibration example.",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=DEFAULT_MIN_TEXT_CHARS,
        help="Skip WikiText rows shorter than this (noise / titles).",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="GPTQ group size (default: 128).",
    )
    parser.add_argument(
        "--damp-percent",
        type=float,
        default=0.01,
        help="GPTQ damp_percent (default: 0.01).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to HF loaders (off by default).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = logging.getLogger("quantize_gptq_3bit")
    _configure_logging(args.verbose)

    root = _repo_root()
    out_dir = (args.output_dir if args.output_dir is not None else root / DEFAULT_OUTPUT_REL).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Model: %s", args.model_id)
    logger.info("Output directory: %s", out_dir)

    _require_cuda(logger)
    torch.cuda.reset_peak_memory_stats()
    _log_vram(logger, "Initial")

    # Baseline on-disk weight size (before/parallel to loading into VRAM).
    try:
        snapshot_dir = _resolve_original_snapshot_dir(args.model_id, logger)
        original_weights_bytes = _weights_disk_bytes(snapshot_dir)
        logger.info(
            "Baseline weight files on disk: %s (%s bytes) under %s",
            _format_gib(original_weights_bytes),
            f"{original_weights_bytes:,}",
            snapshot_dir,
        )
    except Exception as exc:
        logger.exception("Could not measure original model disk size: %s", exc)
        return 1

    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            use_fast=True,
            trust_remote_code=args.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as exc:
        logger.exception("Tokenizer load failed: %s", exc)
        return 1

    # ------------------------------------------------------------------
    # Quantization config: 3-bit weights (AutoGPTQ CUDA path; avoid Triton extras on Windows).
    # ------------------------------------------------------------------
    quantize_config = BaseQuantizeConfig(
        bits=3,
        group_size=args.group_size,
        damp_percent=args.damp_percent,
        desc_act=False,
    )
    logger.info(
        "Quantize config: bits=%s group_size=%s damp_percent=%s desc_act=%s",
        quantize_config.bits,
        quantize_config.group_size,
        quantize_config.damp_percent,
        quantize_config.desc_act,
    )

    examples = _build_calibration_encodings(
        tokenizer,
        num_samples=args.calibration_samples,
        max_seq_len=args.max_seq_len,
        min_chars=args.min_text_chars,
        logger=logger,
    )
    if not examples:
        logger.error("No calibration examples built; aborting.")
        return 1

    try:
        logger.info("Loading base model for quantization (this may take several minutes)...")
        load_kw: dict[str, Any] = dict(
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )
        try:
            model = AutoGPTQForCausalLM.from_pretrained(args.model_id, quantize_config, **load_kw)
        except TypeError as exc:
            logger.warning("Retrying with minimal from_pretrained kwargs (%s)", exc)
            load_kw_min = {k: v for k, v in load_kw.items() if k != "trust_remote_code"}
            try:
                model = AutoGPTQForCausalLM.from_pretrained(args.model_id, quantize_config, **load_kw_min)
            except TypeError:
                model = AutoGPTQForCausalLM.from_pretrained(args.model_id, quantize_config)
    except Exception as exc:
        logger.exception("Failed to load model for quantization: %s", exc)
        return 1

    _log_vram(logger, "After model load")

    peak_quant_gib = 0.0
    try:
        logger.info("Starting GPTQ quantization (3-bit)...")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model.quantize(examples)
        elapsed = time.perf_counter() - t0
        peak_quant_gib = torch.cuda.max_memory_allocated() / (1024**3)
        logger.info("Quantization finished in %.2f s (%.2f min).", elapsed, elapsed / 60.0)
        logger.info("Peak GPU memory during quantize(): %.3f GiB", peak_quant_gib)
    except torch.cuda.OutOfMemoryError:
        logger.exception(
            "CUDA OOM during quantization. Try lowering --max-seq-len, --calibration-samples, "
            "or closing other GPU applications."
        )
        return 1
    except Exception as exc:
        logger.exception("Quantization failed: %s", exc)
        return 1

    _log_vram(logger, "After quantization")

    try:
        logger.info("Saving quantized model to %s ...", out_dir)
        model.save_quantized(str(out_dir), use_safetensors=True)
        tokenizer.save_pretrained(str(out_dir))
    except TypeError:
        logger.warning("save_quantized(use_safetensors=...) not accepted; saving without flag.")
        try:
            model.save_quantized(str(out_dir))
            tokenizer.save_pretrained(str(out_dir))
        except Exception as exc:
            logger.exception("Save failed: %s", exc)
            return 1
    except Exception as exc:
        logger.exception("Save failed: %s", exc)
        return 1

    quantized_weights_bytes = _weights_disk_bytes(out_dir)
    logger.info(
        "Quantized weight files on disk: %s (%s bytes) under %s",
        _format_gib(quantized_weights_bytes),
        f"{quantized_weights_bytes:,}",
        out_dir,
    )

    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    free_b, total_b = torch.cuda.mem_get_info()

    _print_summary_banner(
        logger,
        original_bytes=original_weights_bytes,
        quantized_bytes=quantized_weights_bytes,
        peak_quant_gib=peak_quant_gib,
        post_save_alloc_gib=alloc,
        post_save_reserved_gib=reserved,
        post_save_driver_free_gib=free_b / (1024**3),
        post_save_driver_total_gib=total_b / (1024**3),
    )

    logger.info("Done. Saved GPTQ 3-bit model and tokenizer under: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
