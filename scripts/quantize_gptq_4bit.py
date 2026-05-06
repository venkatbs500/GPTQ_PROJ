#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4-bit GPTQ quantization for TinyLlama-Chat using AutoGPTQ.

Designed for Windows + NVIDIA CUDA. Run from repo root:

    python scripts/quantize_gptq_4bit.py

Or with options:

    python scripts/quantize_gptq_4bit.py --output-dir models/gptq_4bit --calibration-samples 256
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
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


# ---------------------------------------------------------------------------
# Defaults aligned with this repository (TinyLlama Chat, WikiText-2 calibration)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_OUTPUT_REL = Path("models") / "gptq_4bit"
DEFAULT_CALIBRATION_SAMPLES = 256
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_MIN_TEXT_CHARS = 200


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
            "and verify with: python -c \"import torch; print(torch.cuda.is_available())\""
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
    # Stream through the split once; skip empty / very short / section-title lines.
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
        # `return_tensors` omitted: lists on CPU. AutoGPTQ allows only `input_ids` and
        # `attention_mask` in each example dict (see AutoGPTQ Quick Tour).
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

    logger.info("Prepared %d quantized calibration examples (max_seq_len=%d).", len(examples), max_seq_len)
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize TinyLlama-1.1B-Chat with AutoGPTQ (4-bit, safetensors)."
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
    logger = logging.getLogger("quantize_gptq_4bit")
    _configure_logging(args.verbose)

    root = _repo_root()
    out_dir = (args.output_dir if args.output_dir is not None else root / DEFAULT_OUTPUT_REL).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Model: %s", args.model_id)
    logger.info("Output directory: %s", out_dir)

    _require_cuda(logger)
    torch.cuda.reset_peak_memory_stats()
    _log_vram(logger, "Initial")

    # ------------------------------------------------------------------
    # Tokenizer: must be saved alongside weights for inference pipelines.
    # ------------------------------------------------------------------
    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            use_fast=True,
            trust_remote_code=args.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            # Many causal LMs have no pad token; EOS as pad is a common convention.
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as exc:
        logger.exception("Tokenizer load failed: %s", exc)
        return 1

    # ------------------------------------------------------------------
    # Quantization config: 4-bit weights, grouped GPTQ (standard LLM recipe).
    # ------------------------------------------------------------------
    quantize_config = BaseQuantizeConfig(
        bits=4,
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

    # ------------------------------------------------------------------
    # Load model weights, then run GPTQ on GPU.
    # AutoGPTQ 0.7.x Quick Tour uses `from_pretrained(model_id, quantize_config)`; we pass
    # `quantize_config` positionally for widest compatibility, then optional HF kwargs.
    # Do not pass `use_triton` here — that flag applies to `from_quantized`, not calibration.
    # ------------------------------------------------------------------
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
            # Some older forks omit `trust_remote_code` / `device_map` on this entrypoint.
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

    try:
        logger.info("Starting GPTQ quantization...")
        t0 = time.perf_counter()
        model.quantize(examples)
        elapsed = time.perf_counter() - t0
        logger.info("Quantization finished in %.2f seconds (%.2f minutes).", elapsed, elapsed / 60.0)
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

    # ------------------------------------------------------------------
    # Persist weights (safetensors when supported) + tokenizer + config.
    # ------------------------------------------------------------------
    try:
        logger.info("Saving quantized model to %s ...", out_dir)
        model.save_quantized(str(out_dir), use_safetensors=True)
        tokenizer.save_pretrained(str(out_dir))
    except TypeError:
        # Older AutoGPTQ builds may omit use_safetensors; fall back and log.
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

    logger.info("Done. Saved GPTQ 4-bit model and tokenizer under: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
