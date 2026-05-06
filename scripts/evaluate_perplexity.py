#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WikiText-2 perplexity for FP16, GPTQ (3/4-bit), and AWQ checkpoints.

Math (causal LM):
    For tokens x_1..x_T, the model assigns conditional log-probabilities log p(x_t | x_<t).
    The negative log-likelihood (NLL) for one position is -log p (natural log).
    **Mean NLL** over the evaluated positions is the average cross-entropy in nats.
    **Perplexity** is exp(mean NLL): geometric mean of the inverse probabilities.

    Intuition: if mean NLL = ln(100), PPL = 100 (as if the model were uniform over ~100 tokens).

Sliding-window evaluation (stride < max_length) follows Hugging Face's perplexity guide:
    overlapping context windows avoid truncating long documents; loss is accumulated only on
    the *new* trailing tokens each step (mask -100 elsewhere) so tokens are not double-counted.

Run from repo root:

    python scripts/evaluate_perplexity.py

With options:

    python scripts/evaluate_perplexity.py --stride 2048 --batch-size 4
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from auto_gptq import AutoGPTQForCausalLM
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_WIKITEXT = ("wikitext", "wikitext-2-raw-v1", "test")
DEFAULT_MAX_LENGTH = 2048
DEFAULT_STRIDE = 512
DEFAULT_BATCH_SIZE = 1


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
    """AutoAWQ builds vary; try a single-GPU map first, then fall back to ``auto``."""

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


def _load_wikitext_raw_text(logger: logging.Logger, dataset_id: str, config: str, split: str) -> str:
    logger.info("Loading %s / %s split=%s ...", dataset_id, config, split)
    ds = load_dataset(dataset_id, config, split=split)
    parts = [t.strip() for t in ds["text"] if isinstance(t, str) and t.strip()]
    corpus = "\n\n".join(parts)
    logger.info("WikiText raw characters: %s", f"{len(corpus):,}")
    return corpus


@dataclass
class PPLResult:
    name: str
    perplexity: float | None
    eval_seconds: float
    num_loss_tokens: int
    status: str


def _disjoint_windows_ppl(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[float, int]:
    """
    Non-overlapping windows of length ``max_length`` batched together.
    Each window is scored independently (no cross-boundary context). Fast when batch_size > 1.
    """
    seq_len = input_ids.size(1)
    nll_sum = 0.0
    n_tokens = 0
    starts = list(range(0, seq_len - max_length + 1, max_length))
    if not starts:
        raise ValueError(f"Sequence length {seq_len} < max_length {max_length}; increase text or lower --max-length.")

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(starts), batch_size), desc="PPL (batched chunks)"):
            chunk_starts = starts[i : i + batch_size]
            batch = torch.cat(
                [input_ids[:, s : s + max_length] for s in chunk_starts],
                dim=0,
            ).to(device)
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            # Causal LM: loss is mean CE over all (L-1) positions × B sequences.
            n_pos = batch.size(0) * (batch.size(1) - 1)
            nll_sum += float(loss.item()) * n_pos
            n_tokens += n_pos

    mean_nll = nll_sum / n_tokens
    return math.exp(mean_nll), n_tokens


def _sliding_window_ppl(
    model: torch.nn.Module,
    encodings: torch.Tensor,
    *,
    max_length: int,
    stride: int,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[float, int]:
    """
    Hugging Face sliding-window perplexity (batch size 1 along sequence).

    See: https://huggingface.co/docs/transformers/perplexity
    Only the last ``trg_len`` labels in each chunk are supervised; earlier positions are -100
    so each real token's NLL is counted once across overlaps.
    """
    seq_len = encodings.size(1)
    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    model.eval()

    with torch.no_grad():
        for begin_loc in tqdm(range(0, seq_len, stride), desc="PPL (sliding window)"):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

            num_valid = (target_ids != -100).sum().item()
            batch_size = target_ids.size(0)
            # HF note: adjust for internal label shift in CausalLM.
            num_loss_tokens = int(num_valid - batch_size)
            if num_loss_tokens <= 0:
                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
                continue

            nll_sum += float(neg_log_likelihood.item()) * num_loss_tokens
            n_tokens += num_loss_tokens

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    if n_tokens == 0:
        raise RuntimeError("No tokens scored; check max_length / stride / corpus length.")
    mean_nll = nll_sum / n_tokens
    return math.exp(mean_nll), n_tokens


def _try_evaluate_ppl(
    model: torch.nn.Module,
    input_ids_cpu: torch.Tensor,
    *,
    max_length: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[float, int]:
    """Dispatch sliding vs batched disjoint; retry with smaller batch or shorter windows on OOM."""
    if stride >= max_length:
        # Disjoint chunks — can batch along the window dimension.
        bs = batch_size
        while bs >= 1:
            try:
                torch.cuda.empty_cache()
                return _disjoint_windows_ppl(
                    model,
                    input_ids_cpu,
                    max_length=max_length,
                    batch_size=bs,
                    device=device,
                    logger=logger,
                )
            except torch.cuda.OutOfMemoryError:
                logger.warning("CUDA OOM with batch_size=%d; retrying with half.", bs)
                bs //= 2
                if bs < 1:
                    raise
        raise RuntimeError("OOM even at batch_size=1")

    # Overlapping windows — sequential (correct HF recipe).
    torch.cuda.empty_cache()
    try:
        return _sliding_window_ppl(
            model,
            input_ids_cpu,
            max_length=max_length,
            stride=stride,
            device=device,
            logger=logger,
        )
    except torch.cuda.OutOfMemoryError:
        if max_length <= 512:
            raise
        new_len = max(512, max_length // 2)
        new_stride = min(stride, new_len // 2)
        logger.warning(
            "CUDA OOM in sliding window; retrying once with max_length=%d stride=%d.",
            new_len,
            new_stride,
        )
        torch.cuda.empty_cache()
        return _sliding_window_ppl(
            model,
            input_ids_cpu,
            max_length=new_len,
            stride=new_stride,
            device=device,
            logger=logger,
        )


def _print_results_table(results: list[PPLResult], logger: logging.Logger) -> None:
    w_name = max(12, max(len(r.name) for r in results))
    sep = "-" * (w_name + 42)
    lines = [
        sep,
        f"{'Model':<{w_name}}  {'PPL':>12}  {'Tokens':>12}  {'Time (s)':>10}  {'Status':>10}",
        sep,
    ]
    for r in results:
        ppl_s = f"{r.perplexity:.4f}" if r.perplexity is not None and not math.isnan(r.perplexity) else "n/a"
        tok_s = f"{r.num_loss_tokens:,}" if r.num_loss_tokens else "0"
        lines.append(
            f"{r.name:<{w_name}}  {ppl_s:>12}  {tok_s:>12}  {r.eval_seconds:>10.2f}  {r.status:>10}"
        )
    lines.append(sep)
    for line in lines:
        logger.info(line)


def _save_csv(path: Path, results: list[PPLResult], logger: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "perplexity", "eval_seconds", "num_loss_tokens", "status"],
        )
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "model": r.name,
                    "perplexity": "" if r.perplexity is None else f"{r.perplexity:.8f}",
                    "eval_seconds": f"{r.eval_seconds:.4f}",
                    "num_loss_tokens": r.num_loss_tokens,
                    "status": r.status,
                }
            )
    logger.info("Wrote %s", path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WikiText-2 perplexity: FP16 vs GPTQ vs AWQ.")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="HF id for FP16 baseline & tokenizer.")
    p.add_argument("--gptq-4bit-dir", type=Path, default=None, help="Default: <repo>/models/gptq_4bit")
    p.add_argument("--gptq-3bit-dir", type=Path, default=None, help="Default: <repo>/models/gptq_3bit")
    p.add_argument("--awq-4bit-dir", type=Path, default=None, help="Default: <repo>/models/awq_4bit")
    p.add_argument("--wikitext-config", default="wikitext-2-raw-v1")
    p.add_argument("--wikitext-split", default="test")
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Sliding stride; set >= --max-length for batched disjoint windows.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Only used when stride >= max_length (disjoint batching).")
    p.add_argument("--output-csv", type=Path, default=None, help="Default: <repo>/results/csv/perplexity_results.csv")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger = logging.getLogger("evaluate_perplexity")
    _configure_logging(args.verbose)

    root = _repo_root()
    device = torch.device("cuda:0")
    _require_cuda(logger)

    gptq4 = args.gptq_4bit_dir or (root / "models" / "gptq_4bit")
    gptq3 = args.gptq_3bit_dir or (root / "models" / "gptq_3bit")
    awq4 = args.awq_4bit_dir or (root / "models" / "awq_4bit")
    awq_cls = _get_awq_class(logger)
    out_csv = args.output_csv or (root / "results" / "csv" / "perplexity_results.csv")

    corpus = _load_wikitext_raw_text(logger, "wikitext", args.wikitext_config, args.wikitext_split)

    logger.info("Loading tokenizer from %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(corpus, return_tensors="pt", add_special_tokens=False)
    input_ids_cpu = enc["input_ids"]
    logger.info("Tokenized length: %s tokens", f"{input_ids_cpu.size(1):,}")

    # Each entry: (display_name, loader callable -> model)
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

    results: list[PPLResult] = []

    for name, loader in jobs:
        logger.info("=== Evaluating: %s ===", name)
        t0 = time.perf_counter()
        try:
            model = loader()
        except Exception as exc:
            logger.exception("Load failed: %s", exc)
            results.append(
                PPLResult(
                    name=name,
                    perplexity=None,
                    eval_seconds=time.perf_counter() - t0,
                    num_loss_tokens=0,
                    status="load_err",
                )
            )
            continue

        try:
            ppl, n_tok = _try_evaluate_ppl(
                model,
                input_ids_cpu,
                max_length=args.max_length,
                stride=args.stride,
                batch_size=max(1, args.batch_size),
                device=device,
                logger=logger,
            )
            elapsed = time.perf_counter() - t0
            logger.info("%s | perplexity=%.4f | tokens=%s | time=%.2fs", name, ppl, f"{n_tok:,}", elapsed)
            results.append(
                PPLResult(
                    name=name,
                    perplexity=ppl,
                    eval_seconds=elapsed,
                    num_loss_tokens=n_tok,
                    status="ok",
                )
            )
        except torch.cuda.OutOfMemoryError:
            elapsed = time.perf_counter() - t0
            logger.exception("CUDA OOM during %s", name)
            results.append(
                PPLResult(
                    name=name,
                    perplexity=None,
                    eval_seconds=elapsed,
                    num_loss_tokens=0,
                    status="oom",
                )
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.exception("Eval failed: %s", exc)
            results.append(
                PPLResult(
                    name=name,
                    perplexity=None,
                    eval_seconds=elapsed,
                    num_loss_tokens=0,
                    status="eval_err",
                )
            )
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    _print_results_table(results, logger)
    _save_csv(out_csv.resolve(), results, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
