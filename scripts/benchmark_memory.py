#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU memory footprint for FP16 vs GPTQ vs AWQ checkpoints (PyTorch CUDA allocator).

What the numbers mean (PyTorch CUDA caching allocator)
------------------------------------------------------
**memory_allocated(device)** — Total size in bytes of **live** tensors the allocator has
handed out on that device. This is the memory your model weights, activations, and
buffers currently "own" from PyTorch's perspective.

**memory_reserved(device)** — Total CUDA memory **reserved** by the caching allocator
from the driver. Reserved ≥ allocated: PyTorch keeps pools of free blocks to reduce
``cudaMalloc`` churn, so reserved memory can stay high briefly even after tensors are freed.

**max_memory_allocated(device)** — **High-water mark** of ``memory_allocated`` since the
last ``torch.cuda.reset_peak_memory_stats()``. Use this as **peak allocated** during a
phase (e.g. model load or one ``generate``) after resetting peaks at a known boundary.

**Synchronization:** ``torch.cuda.synchronize()`` before reading stats ensures queued
kernel memory (e.g. temporary buffers) is reflected in allocator state for a fair read.

This script reports GiB (binary gibibytes, 1024³) for readability.

Phases
------
1. Reset peaks → **load** model → snapshot (**load_*** columns).
2. Run **one** greedy ``generate()`` (same prompt/max_new_tokens as other benchmarks) →
   snapshot (**post_gen_*** columns). Peak after generation includes any KV / activation
   spikes; ``peak_delta_gen_gib`` approximates extra high-water mark after load.

Run from repo root::

    python scripts/benchmark_memory.py
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPT = (
    "You are a helpful assistant. Explain in three sentences what quantization does "
    "for large language models and why it speeds up inference on GPUs."
)
DEFAULT_MAX_NEW_TOKENS = 64
CUDA_DEVICE_INDEX = 0


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


def _has_gptq_artifacts(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "quantize_config.json").exists():
        return True
    return bool(any(path.glob("*.safetensors")) or any(path.glob("*.bin")))


def _has_awq_artifacts(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "config.json").exists():
        return True
    return bool(any(path.glob("*.safetensors")))


def _get_awq_class(logger: logging.Logger):
    """Best-effort AWQ import. Returns None when AWQ is unavailable."""
    try:
        mod = importlib.import_module("awq")
        return mod.AutoAWQForCausalLM
    except Exception as exc:
        logger.warning("AWQ unavailable; skipping AWQ runs (%s).", exc)
        return None


def _make_awq_loader(path: Path, trust_remote_code: bool, awq_cls) -> Callable[[], torch.nn.Module]:
    def _load() -> torch.nn.Module:
        try:
            return awq_cls.from_quantized(
                str(path),
                fuse_layers=True,
                device_map={"": "cuda:0"},
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            return awq_cls.from_quantized(
                str(path),
                fuse_layers=True,
                device_map="auto",
                trust_remote_code=trust_remote_code,
            )

    return _load


def _require_cuda(logger: logging.Logger) -> None:
    if not torch.cuda.is_available():
        logger.error("CUDA is required for this script.")
        raise SystemExit(2)
    logger.info("GPU: %s", torch.cuda.get_device_name(CUDA_DEVICE_INDEX))


def _infer_device(model: torch.nn.Module) -> torch.device:
    dev = getattr(model, "device", None)
    if dev is not None:
        return dev if isinstance(dev, torch.device) else torch.device(dev)
    p = next(model.parameters(), None)
    if p is not None:
        return p.device
    return torch.device("cuda", CUDA_DEVICE_INDEX)


def _snapshot_gib() -> tuple[float, float, float]:
    """(allocated_gib, reserved_gib, max_allocated_gib) on CUDA_DEVICE_INDEX."""
    idx = CUDA_DEVICE_INDEX
    torch.cuda.synchronize(idx)
    a = torch.cuda.memory_allocated(idx) / (1024**3)
    r = torch.cuda.memory_reserved(idx) / (1024**3)
    p = torch.cuda.max_memory_allocated(idx) / (1024**3)
    return float(a), float(r), float(p)


@dataclass
class MemResult:
    name: str
    load_alloc_gib: float | None
    load_reserved_gib: float | None
    load_peak_gib: float | None
    post_gen_alloc_gib: float | None
    post_gen_reserved_gib: float | None
    post_gen_peak_gib: float | None
    peak_delta_gen_gib: float | None
    status: str


def _print_load_table(rows: list[MemResult], logger: logging.Logger) -> None:
    wn = max(14, max(len(r.name) for r in rows))
    sep = "-" * (wn + 52)
    hdr = f"{'Model':<{wn}}  {'alloc':>10}  {'reserved':>10}  {'peak':>10}  {'Status':>8}"
    lines = [sep, "After model load (GiB)", sep, hdr, sep]
    for r in rows:
        if r.load_alloc_gib is None:
            a, rs, pk = "n/a", "n/a", "n/a"
        else:
            a, rs, pk = f"{r.load_alloc_gib:.3f}", f"{r.load_reserved_gib:.3f}", f"{r.load_peak_gib:.3f}"
        lines.append(f"{r.name:<{wn}}  {a:>10}  {rs:>10}  {pk:>10}  {r.status:>8}")
    lines.append(sep)
    for ln in lines:
        logger.info(ln)


def _print_gen_table(rows: list[MemResult], logger: logging.Logger) -> None:
    wn = max(14, max(len(r.name) for r in rows))
    sep = "-" * (wn + 62)
    hdr = (
        f"{'Model':<{wn}}  {'alloc':>10}  {'reserved':>10}  {'peak':>10}  "
        f"{'Δpeak(gen)':>12}  {'Status':>8}"
    )
    lines = [sep, "After 1× generate() (GiB)", sep, hdr, sep]
    for r in rows:
        if r.post_gen_alloc_gib is None:
            a, rs, pk, d = "n/a", "n/a", "n/a", "n/a"
        else:
            a = f"{r.post_gen_alloc_gib:.3f}"
            rs = f"{r.post_gen_reserved_gib:.3f}"
            pk = f"{r.post_gen_peak_gib:.3f}"
            d = f"{r.peak_delta_gen_gib:.3f}" if r.peak_delta_gen_gib is not None else "n/a"
        lines.append(f"{r.name:<{wn}}  {a:>10}  {rs:>10}  {pk:>10}  {d:>12}  {r.status:>8}")
    lines.append(sep)
    for ln in lines:
        logger.info(ln)


def _save_csv(path: Path, rows: list[MemResult], logger: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "load_alloc_gib",
        "load_reserved_gib",
        "load_peak_allocated_gib",
        "post_gen_alloc_gib",
        "post_gen_reserved_gib",
        "post_gen_peak_allocated_gib",
        "peak_delta_gen_gib",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            def fmt(x: float | None) -> str:
                return "" if x is None else f"{x:.8f}"

            w.writerow(
                {
                    "model": r.name,
                    "load_alloc_gib": fmt(r.load_alloc_gib),
                    "load_reserved_gib": fmt(r.load_reserved_gib),
                    "load_peak_allocated_gib": fmt(r.load_peak_gib),
                    "post_gen_alloc_gib": fmt(r.post_gen_alloc_gib),
                    "post_gen_reserved_gib": fmt(r.post_gen_reserved_gib),
                    "post_gen_peak_allocated_gib": fmt(r.post_gen_peak_gib),
                    "peak_delta_gen_gib": fmt(r.peak_delta_gen_gib),
                    "status": r.status,
                }
            )
    logger.info("Wrote %s", path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU memory: FP16 vs GPTQ vs AWQ.")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--gptq-4bit-dir", type=Path, default=None)
    p.add_argument("--gptq-3bit-dir", type=Path, default=None)
    p.add_argument("--awq-4bit-dir", type=Path, default=None)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--prompt-file", type=Path, default=None)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--skip-generate",
        action="store_true",
        help="Only measure memory after load (no generate pass).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger = logging.getLogger("benchmark_memory")
    _configure_logging(args.verbose)
    _require_cuda(logger)

    root = _repo_root()
    gptq4 = args.gptq_4bit_dir or (root / "models" / "gptq_4bit")
    gptq3 = args.gptq_3bit_dir or (root / "models" / "gptq_3bit")
    awq4 = args.awq_4bit_dir or (root / "models" / "awq_4bit")
    awq_cls = _get_awq_class(logger)
    out_csv = args.output_csv or (root / "results" / "csv" / "memory_results.csv")

    prompt = args.prompt
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text(encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")

    gen_kwargs: dict = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    jobs: list[tuple[str, Callable[[], torch.nn.Module]]] = [
        (
            "FP16 (original)",
            lambda: AutoModelForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                trust_remote_code=args.trust_remote_code,
            ),
        )
    ]

    if _has_gptq_artifacts(gptq4):
        jobs.append(
            (
                "GPTQ 4-bit",
                lambda p=gptq4: AutoGPTQForCausalLM.from_quantized(
                    str(p),
                    device="cuda:0",
                    use_triton=False,
                    trust_remote_code=args.trust_remote_code,
                ),
            )
        )
    else:
        logger.warning("Skipping GPTQ 4-bit (no artifacts under %s).", gptq4)

    if _has_gptq_artifacts(gptq3):
        jobs.append(
            (
                "GPTQ 3-bit",
                lambda p=gptq3: AutoGPTQForCausalLM.from_quantized(
                    str(p),
                    device="cuda:0",
                    use_triton=False,
                    trust_remote_code=args.trust_remote_code,
                ),
            )
        )
    else:
        logger.warning("Skipping GPTQ 3-bit (no artifacts under %s).", gptq3)

    if awq_cls is not None and _has_awq_artifacts(awq4):
        jobs.append(("AWQ 4-bit", _make_awq_loader(awq4, args.trust_remote_code, awq_cls)))
    elif awq_cls is None:
        logger.info("Skipping AWQ 4-bit (package not installed).")
    else:
        logger.warning("Skipping AWQ 4-bit (no artifacts under %s).", awq4)

    results: list[MemResult] = []

    for name, loader in jobs:
        logger.info("=== Memory benchmark: %s ===", name)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(CUDA_DEVICE_INDEX)

        try:
            model = loader()
        except Exception as exc:
            logger.exception("Load failed: %s", exc)
            results.append(
                MemResult(
                    name=name,
                    load_alloc_gib=None,
                    load_reserved_gib=None,
                    load_peak_gib=None,
                    post_gen_alloc_gib=None,
                    post_gen_reserved_gib=None,
                    post_gen_peak_gib=None,
                    peak_delta_gen_gib=None,
                    status="load_err",
                )
            )
            continue

        model.eval()
        device = _infer_device(model)
        input_ids_d = input_ids.to(device)
        attn = attention_mask.to(device) if attention_mask is not None else None

        try:
            torch.cuda.synchronize(CUDA_DEVICE_INDEX)
            la, lr, lp = _snapshot_gib()
            peak_after_load = lp

            pg_a = pg_r = pg_p = delta = None
            if not args.skip_generate:
                with torch.no_grad():
                    if attn is not None:
                        _ = model.generate(
                            input_ids=input_ids_d,
                            attention_mask=attn,
                            **gen_kwargs,
                        )
                    else:
                        _ = model.generate(input_ids=input_ids_d, **gen_kwargs)
                torch.cuda.synchronize(CUDA_DEVICE_INDEX)
                pg_a, pg_r, pg_p = _snapshot_gib()
                delta = pg_p - peak_after_load

            results.append(
                MemResult(
                    name=name,
                    load_alloc_gib=la,
                    load_reserved_gib=lr,
                    load_peak_gib=lp,
                    post_gen_alloc_gib=pg_a,
                    post_gen_reserved_gib=pg_r,
                    post_gen_peak_gib=pg_p,
                    peak_delta_gen_gib=delta,
                    status="ok",
                )
            )
        except torch.cuda.OutOfMemoryError:
            logger.exception("CUDA OOM: %s", name)
            results.append(
                MemResult(
                    name=name,
                    load_alloc_gib=None,
                    load_reserved_gib=None,
                    load_peak_gib=None,
                    post_gen_alloc_gib=None,
                    post_gen_reserved_gib=None,
                    post_gen_peak_gib=None,
                    peak_delta_gen_gib=None,
                    status="oom",
                )
            )
        except Exception as exc:
            logger.exception("Benchmark failed: %s", exc)
            results.append(
                MemResult(
                    name=name,
                    load_alloc_gib=None,
                    load_reserved_gib=None,
                    load_peak_gib=None,
                    post_gen_alloc_gib=None,
                    post_gen_reserved_gib=None,
                    post_gen_peak_gib=None,
                    peak_delta_gen_gib=None,
                    status="eval_err",
                )
            )
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    _print_load_table(results, logger)
    if not args.skip_generate:
        _print_gen_table(results, logger)
    _save_csv(out_csv.resolve(), results, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
