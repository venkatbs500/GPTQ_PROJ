#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoregressive generation speed benchmark: FP16 vs GPTQ vs AWQ (CUDA).

Methodology (timing):
    PyTorch CUDA kernels launch asynchronously: ``time.perf_counter()`` around
    ``model.generate()`` *without* synchronization mostly measures CPU enqueue time.
    This script calls ``torch.cuda.synchronize()`` **immediately before** starting
    the timer and **immediately after** ``generate`` returns so each timed interval
    reflects **completed GPU work** for that call (wall time on the critical path).

    Warmup iterations (default 3) are **excluded** from statistics so CUDA context,
    cuBLAS workspaces, and kernel caches do not skew averages.

Metrics:
    - **avg_runtime_s**: mean wall time per ``generate()`` call (full prompt + decoding).
    - **latency_ms**: same as avg runtime, in milliseconds (end-to-end completion latency).
    - **tokens_per_sec**: ``mean(new_tokens) / avg_runtime_s`` — decode throughput using
      *actual* newly generated tokens (may be < ``max_new_tokens`` if EOS fires early).
    - **peak_gpu_gib**: ``torch.cuda.max_memory_allocated()`` peak after timed loop
      (allocator high-water mark on the active device).

All models use the **same** tokenized prompt and identical ``GenerationConfig``-equivalent
arguments so comparisons are apples-to-apples.

Run from repo root:

    python scripts/benchmark_speed.py

    python scripts/benchmark_speed.py --max-new-tokens 128 --iterations 20 --warmup 5
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import logging
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from auto_gptq import AutoGPTQForCausalLM
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPT = (
    "You are a helpful assistant. Explain in three sentences what quantization does "
    "for large language models and why it speeds up inference on GPUs."
)
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_WARMUP = 3
DEFAULT_ITERATIONS = 10


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
    logger.info("GPU: %s", torch.cuda.get_device_name(0))


def _infer_device(model: torch.nn.Module) -> torch.device:
    """Resolve device for input tensors (handles ``device_map='auto'`` / quantized wrappers)."""
    dev = getattr(model, "device", None)
    if dev is not None:
        return dev if isinstance(dev, torch.device) else torch.device(dev)
    p = next(model.parameters(), None)
    if p is not None:
        return p.device
    return torch.device("cuda:0")


@dataclass
class SpeedResult:
    name: str
    avg_runtime_s: float | None
    latency_ms: float | None
    tokens_per_sec: float | None
    avg_new_tokens: float | None
    peak_gpu_gib: float | None
    warmup_runs: int
    timed_runs: int
    status: str


def _benchmark_generate(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    max_new_tokens: int,
    warmup: int,
    iterations: int,
    logger: logging.Logger,
) -> tuple[float, float, float, float]:
    """
    Returns (avg_runtime_s, mean_new_tokens, tokens_per_sec, peak_gpu_gib).
    """
    device = _infer_device(model)
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    model.eval()
    prompt_len = int(input_ids.shape[1])

    def _generate() -> torch.Tensor:
        if attention_mask is not None:
            return model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        return model.generate(input_ids=input_ids, **gen_kwargs)

    with torch.no_grad():
        for w in range(warmup):
            torch.cuda.synchronize()
            _ = _generate()
            torch.cuda.synchronize()
            logger.debug("Warmup %d/%d done", w + 1, warmup)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    runtimes: list[float] = []
    new_counts: list[int] = []

    for _ in tqdm(range(iterations), desc="Timed generations", unit="run"):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = _generate()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        elapsed = t1 - t0
        new_tok = int(out.shape[1] - prompt_len)
        runtimes.append(elapsed)
        new_counts.append(new_tok)

    avg_time = float(statistics.mean(runtimes))
    mean_new = float(statistics.mean(new_counts))
    tps = mean_new / avg_time if avg_time > 0 else 0.0
    peak = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    return avg_time, mean_new, tps, float(peak)


def _print_table(rows: list[SpeedResult], logger: logging.Logger) -> None:
    wn = max(14, max(len(r.name) for r in rows))
    sep = "-" * (wn + 86)
    head = (
        f"{'Model':<{wn}}  {'t/s':>10}  {'lat(ms)':>10}  {'avgRun(s)':>10}  "
        f"{'newTok':>8}  {'peakGiB':>8}  {'Status':>8}"
    )
    lines = [sep, head, sep]
    for r in rows:
        ts = f"{r.tokens_per_sec:.2f}" if r.tokens_per_sec is not None else "n/a"
        lm = f"{r.latency_ms:.2f}" if r.latency_ms is not None else "n/a"
        ar = f"{r.avg_runtime_s:.4f}" if r.avg_runtime_s is not None else "n/a"
        nt = f"{r.avg_new_tokens:.1f}" if r.avg_new_tokens is not None else "n/a"
        pg = f"{r.peak_gpu_gib:.2f}" if r.peak_gpu_gib is not None else "n/a"
        lines.append(f"{r.name:<{wn}}  {ts:>10}  {lm:>10}  {ar:>10}  {nt:>8}  {pg:>8}  {r.status:>8}")
    lines.append(sep)
    for ln in lines:
        logger.info(ln)


def _save_csv(path: Path, rows: list[SpeedResult], logger: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "tokens_per_sec",
        "latency_ms",
        "avg_runtime_s",
        "avg_new_tokens",
        "peak_gpu_gib",
        "warmup_runs",
        "timed_runs",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "model": r.name,
                    "tokens_per_sec": "" if r.tokens_per_sec is None else f"{r.tokens_per_sec:.6f}",
                    "latency_ms": "" if r.latency_ms is None else f"{r.latency_ms:.6f}",
                    "avg_runtime_s": "" if r.avg_runtime_s is None else f"{r.avg_runtime_s:.8f}",
                    "avg_new_tokens": "" if r.avg_new_tokens is None else f"{r.avg_new_tokens:.4f}",
                    "peak_gpu_gib": "" if r.peak_gpu_gib is None else f"{r.peak_gpu_gib:.6f}",
                    "warmup_runs": r.warmup_runs,
                    "timed_runs": r.timed_runs,
                    "status": r.status,
                }
            )
    logger.info("Wrote %s", path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference speed: FP16 vs GPTQ vs AWQ.")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--gptq-4bit-dir", type=Path, default=None)
    p.add_argument("--gptq-3bit-dir", type=Path, default=None)
    p.add_argument("--awq-4bit-dir", type=Path, default=None)
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="Single shared prompt for all models.")
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="If set, overrides --prompt with file contents (UTF-8).",
    )
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger = logging.getLogger("benchmark_speed")
    _configure_logging(args.verbose)
    if args.iterations < 1:
        logger.error("--iterations must be >= 1.")
        return 2
    if args.warmup < 0:
        logger.error("--warmup must be >= 0.")
        return 2
    _require_cuda(logger)

    root = _repo_root()
    gptq4 = args.gptq_4bit_dir or (root / "models" / "gptq_4bit")
    gptq3 = args.gptq_3bit_dir or (root / "models" / "gptq_3bit")
    awq4 = args.awq_4bit_dir or (root / "models" / "awq_4bit")
    awq_cls = _get_awq_class(logger)
    out_csv = args.output_csv or (root / "results" / "csv" / "speed_results.csv")

    prompt = args.prompt
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    logger.info("Prompt chars: %d | max_new_tokens=%d", len(prompt), args.max_new_tokens)

    logger.info("Loading tokenizer from %s", args.base_model)
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
    if attention_mask is not None:
        attention_mask = attention_mask  # keep for generate()
    logger.info("Prompt tokens: %d", input_ids.shape[1])

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

    results: list[SpeedResult] = []

    for name, loader in jobs:
        logger.info("=== Benchmark: %s ===", name)
        try:
            model = loader()
        except Exception as exc:
            logger.exception("Load failed: %s", exc)
            results.append(
                SpeedResult(
                    name=name,
                    avg_runtime_s=None,
                    latency_ms=None,
                    tokens_per_sec=None,
                    avg_new_tokens=None,
                    peak_gpu_gib=None,
                    warmup_runs=args.warmup,
                    timed_runs=args.iterations,
                    status="load_err",
                )
            )
            continue

        try:
            avg_s, mean_new, tps, peak = _benchmark_generate(
                model,
                tokenizer,
                input_ids,
                attention_mask,
                max_new_tokens=args.max_new_tokens,
                warmup=args.warmup,
                iterations=args.iterations,
                logger=logger,
            )
            results.append(
                SpeedResult(
                    name=name,
                    avg_runtime_s=avg_s,
                    latency_ms=avg_s * 1000.0,
                    tokens_per_sec=tps,
                    avg_new_tokens=mean_new,
                    peak_gpu_gib=peak,
                    warmup_runs=args.warmup,
                    timed_runs=args.iterations,
                    status="ok",
                )
            )
            logger.info(
                "%s | %.2f tok/s | latency %.2f ms | peak %.2f GiB",
                name,
                tps,
                avg_s * 1000.0,
                peak,
            )
        except torch.cuda.OutOfMemoryError:
            logger.exception("CUDA OOM: %s", name)
            results.append(
                SpeedResult(
                    name=name,
                    avg_runtime_s=None,
                    latency_ms=None,
                    tokens_per_sec=None,
                    avg_new_tokens=None,
                    peak_gpu_gib=None,
                    warmup_runs=args.warmup,
                    timed_runs=args.iterations,
                    status="oom",
                )
            )
        except Exception as exc:
            logger.exception("Benchmark failed: %s", exc)
            results.append(
                SpeedResult(
                    name=name,
                    avg_runtime_s=None,
                    latency_ms=None,
                    tokens_per_sec=None,
                    avg_new_tokens=None,
                    peak_gpu_gib=None,
                    warmup_runs=args.warmup,
                    timed_runs=args.iterations,
                    status="eval_err",
                )
            )
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    _print_table(results, logger)
    _save_csv(out_csv.resolve(), results, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
