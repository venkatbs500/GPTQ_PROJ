#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4-bit AWQ quantization for TinyLlama-Chat using AutoAWQ.

AWQ workflow (high level):
    1) Load full-precision (typically FP16) weights on GPU.
    2) Run calibration: feed representative text through the model to estimate
       activation scales. AWQ is *activation-aware* (unlike basic GPTQ), which
       helps preserve quality at low bit widths.
    3) Solve per-channel (or per-group) scaling and pack weights into INT4,
       optionally with zero points, using fused GEMM kernels (``version: GEMM``).
    4) Save packed weights + config; reload later via ``from_quantized`` for inference.

Designed for Windows + NVIDIA CUDA. Run from repo root:

    python scripts/quantize_awq_4bit.py

With options:

    python scripts/quantize_awq_4bit.py --max-calib-samples 256 --max-calib-seq-len 512
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Defaults: TinyLlama Chat + WikiText-2 calibration (list[str] for AutoAWQ)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_OUTPUT_REL = Path("models") / "awq_4bit"
DEFAULT_CALIB_TEXTS = 256
DEFAULT_MIN_TEXT_CHARS = 200
# AutoAWQ defaults to max_calib_seq_len=512; longer snippets are skipped unless raised.
DEFAULT_MAX_CALIB_SEQ_LEN = 512


def _import_autoawq_for_tinyllama(logger: logging.Logger):
    """
    Import AutoAWQ with a local compatibility shim for transformers==4.37.2.

    Some AutoAWQ builds import ``transformers.models.gemma.modeling_gemma`` at
    module import time, but Gemma modules were introduced in later transformers
    releases. For TinyLlama (LLaMA family), Gemma code paths are unused; we add a
    minimal local shim so the import succeeds without upgrading transformers.
    """
    try:
        from awq import AutoAWQForCausalLM
        return AutoAWQForCausalLM
    except ModuleNotFoundError as exc:
        if "transformers.models.gemma" not in str(exc):
            raise

    logger.warning(
        "Applying local AutoAWQ compatibility shim for missing Gemma module "
        "(transformers==4.37.2)."
    )
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    gemma_pkg = types.ModuleType("transformers.models.gemma")
    gemma_mod = types.ModuleType("transformers.models.gemma.modeling_gemma")
    # TinyLlama path never uses Gemma layers; alias keeps AutoAWQ importable.
    gemma_mod.GemmaRMSNorm = LlamaRMSNorm
    sys.modules["transformers.models.gemma"] = gemma_pkg
    sys.modules["transformers.models.gemma.modeling_gemma"] = gemma_mod

    from awq import AutoAWQForCausalLM

    return AutoAWQForCausalLM


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _require_cuda(logger: logging.Logger) -> torch.device:
    """AutoAWQ CUDA kernels expect an NVIDIA GPU; fail fast on CPU-only setups."""
    if not torch.cuda.is_available():
        logger.error(
            "CUDA is not available. Install a CUDA-enabled PyTorch build (see README) "
            'and verify with: python -c "import torch; print(torch.cuda.is_available())"'
        )
        raise SystemExit(2)
    device = torch.device("cuda:0")
    logger.info("Using GPU: %s (cuda:%s)", torch.cuda.get_device_name(device), device.index)
    return device


def _log_vram(logger: logging.Logger, label: str) -> None:
    """Driver free/total + PyTorch allocator stats (works on Windows)."""
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


def _build_calib_texts(
    *,
    num_texts: int,
    min_chars: int,
    logger: logging.Logger,
) -> list[str]:
    """
    Build calibration *plain text* lines for AutoAWQ.

    ``model.quantize(..., calib_data=...)`` expects a ``list[str]``; the library
    tokenizes internally. WikiText-2 raw is small and standard for LM calibration.
    """
    logger.info(
        "Loading WikiText-2 (wikitext-2-raw-v1 train) for up to %d calibration texts...",
        num_texts,
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
        if len(texts) >= num_texts:
            break

    if len(texts) < num_texts:
        logger.warning(
            "Only collected %d texts (wanted %d). Proceeding with available data.",
            len(texts),
            num_texts,
        )
    logger.info("Prepared %d calibration text snippets.", len(texts))
    return texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize TinyLlama-1.1B-Chat with AutoAWQ (4-bit, GEMM, safetensors)."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id or local path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: <repo>/{DEFAULT_OUTPUT_REL.as_posix()}).",
    )
    parser.add_argument(
        "--calibration-texts",
        type=int,
        default=DEFAULT_CALIB_TEXTS,
        help="Target number of WikiText snippets to collect before AutoAWQ subsampling.",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=DEFAULT_MIN_TEXT_CHARS,
        help="Skip WikiText rows shorter than this.",
    )
    parser.add_argument(
        "--max-calib-samples",
        type=int,
        default=128,
        help="AutoAWQ max_calib_samples (default 128; raise if you need more coverage).",
    )
    parser.add_argument(
        "--max-calib-seq-len",
        type=int,
        default=DEFAULT_MAX_CALIB_SEQ_LEN,
        help="AutoAWQ max_calib_seq_len (token cap per sample; default 512).",
    )
    parser.add_argument(
        "--n-parallel-calib-samples",
        type=int,
        default=None,
        help="Optional AutoAWQ n_parallel_calib_samples (RAM/VRAM tradeoff; omit to use library default).",
    )
    parser.add_argument(
        "--q-group-size",
        type=int,
        default=128,
        help="AWQ group size (default 128).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Debug logging.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to HF loaders.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = logging.getLogger("quantize_awq_4bit")
    _configure_logging(args.verbose)
    try:
        AutoAWQForCausalLM = _import_autoawq_for_tinyllama(logger)
    except Exception as exc:
        logger.exception("Failed to import AutoAWQ: %s", exc)
        return 1

    root = _repo_root()
    out_dir = (args.output_dir if args.output_dir is not None else root / DEFAULT_OUTPUT_REL).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Model: %s", args.model_id)
    logger.info("Output directory: %s", out_dir)

    _require_cuda(logger)
    torch.cuda.reset_peak_memory_stats()
    _log_vram(logger, "Initial")

    # ------------------------------------------------------------------
    # Tokenizer: saved next to weights for inference / evaluation pipelines.
    # ------------------------------------------------------------------
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

    calib_texts = _build_calib_texts(
        num_texts=args.calibration_texts,
        min_chars=args.min_text_chars,
        logger=logger,
    )
    if not calib_texts:
        logger.error("No calibration texts; aborting.")
        return 1

    # ------------------------------------------------------------------
    # AWQ quant config: 4-bit weights, grouped, GEMM backend (Windows-friendly CUDA).
    # ``zero_point=True`` is the common default in AutoAWQ examples.
    # ------------------------------------------------------------------
    quant_config: dict[str, Any] = {
        "zero_point": True,
        "q_group_size": args.q_group_size,
        "w_bit": 4,
        "version": "GEMM",
    }
    logger.info("Quant config: %s", quant_config)

    # ------------------------------------------------------------------
    # Load FP model: FP16 on GPU via accelerate ``device_map`` (standard on Windows).
    # ------------------------------------------------------------------
    try:
        logger.info("Loading model for AWQ calibration (may take a few minutes)...")
        model = AutoAWQForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
            safetensors=True,
        )
    except Exception as exc:
        logger.exception("from_pretrained failed: %s", exc)
        return 1

    _log_vram(logger, "After model load")

    # ------------------------------------------------------------------
    # Quantize: runs AWQ calibration + weight packing on CUDA.
    # Optional kwargs tune VRAM vs speed (see AutoAWQ docs).
    # ------------------------------------------------------------------
    quantize_kw: dict[str, Any] = {
        "quant_config": quant_config,
        "calib_data": calib_texts,
        "max_calib_samples": args.max_calib_samples,
        "max_calib_seq_len": args.max_calib_seq_len,
    }
    if args.n_parallel_calib_samples is not None:
        quantize_kw["n_parallel_calib_samples"] = args.n_parallel_calib_samples

    peak_quant_gib = 0.0
    try:
        logger.info("Starting AWQ quantization...")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        # Pass tokenizer positionally — matches AutoAWQ examples and older signatures.
        model.quantize(tokenizer, **quantize_kw)
        elapsed = time.perf_counter() - t0
        peak_quant_gib = torch.cuda.max_memory_allocated() / (1024**3)
        logger.info("Quantization finished in %.2f s (%.2f min).", elapsed, elapsed / 60.0)
        logger.info("Peak GPU memory during quantize(): %.3f GiB", peak_quant_gib)
    except torch.cuda.OutOfMemoryError:
        logger.exception(
            "CUDA OOM. Try lowering --max-calib-seq-len, --max-calib-samples, "
            "or set --n-parallel-calib-samples to offload calibration (uses system RAM)."
        )
        return 1
    except Exception as exc:
        logger.exception("Quantization failed: %s", exc)
        return 1

    _log_vram(logger, "After quantization")

    # ------------------------------------------------------------------
    # Persist packed INT4 weights (safetensors) + tokenizer files.
    # ------------------------------------------------------------------
    try:
        logger.info("Saving quantized model to %s ...", out_dir)
        model.save_quantized(str(out_dir), safetensors=True)
        tokenizer.save_pretrained(str(out_dir))
    except TypeError:
        logger.warning("save_quantized(safetensors=...) not accepted; saving with defaults.")
        try:
            model.save_quantized(str(out_dir))
            tokenizer.save_pretrained(str(out_dir))
        except Exception as exc:
            logger.exception("Save failed: %s", exc)
            return 1
    except Exception as exc:
        logger.exception("Save failed: %s", exc)
        return 1

    _log_vram(logger, "After save")
    logger.info("Done. AWQ 4-bit model and tokenizer saved under: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
