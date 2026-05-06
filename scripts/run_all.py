#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-command experiment pipeline (Windows-friendly).

Runs, in order:
    1. GPTQ 4-bit quantization     → ``quantize_gptq_4bit.py``
    2. GPTQ 3-bit quantization     → ``quantize_gptq_3bit.py``
    3. AWQ 4-bit quantization      → ``quantize_awq_4bit.py``
    4. WikiText-2 perplexity       → ``evaluate_perplexity.py``
    5. Inference speed             → ``benchmark_speed.py``
    6. GPU memory                  → ``benchmark_memory.py``
    7. Figures from CSVs           → ``generate_graphs.py``

Reproducibility: uses ``sys.executable`` and repository root as ``cwd`` for every subprocess.
Set ``PYTHONUNBUFFERED=1`` so logs flush promptly on Windows consoles.

Example (PowerShell, repo root)::

    python scripts/run_all.py

    python scripts/run_all.py --continue-on-error --skip-quantization

    python scripts/run_all.py --dry-run

    python scripts/run_all.py --verify-env --calibration-samples 256
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _ensure_directories(root: Path, logger: logging.Logger) -> None:
    """Create artifact and results folders before child scripts run."""
    rels = [
        Path("models") / "gptq_4bit",
        Path("models") / "gptq_3bit",
        Path("models") / "awq_4bit",
        Path("results") / "csv",
        Path("results") / "figures",
    ]
    for rel in rels:
        p = root / rel
        p.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory: %s", p)


@dataclass
class StepRecord:
    name: str
    script: str
    returncode: int | None = None
    seconds: float = 0.0
    skipped: bool = False


def _run_subprocess(
    root: Path,
    script_rel: str,
    extra_args: list[str],
    logger: logging.Logger,
    dry_run: bool,
) -> tuple[int, float]:
    script_path = root / script_rel
    if not script_path.is_file():
        logger.error("Script not found: %s", script_path)
        return 127, 0.0

    cmd = [sys.executable, str(script_path)] + extra_args
    logger.info("Command: %s", subprocess.list2cmdline(cmd))

    if dry_run:
        return 0, 0.0

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        env=env,
        stdin=subprocess.DEVNULL,
    )
    elapsed = time.perf_counter() - t0
    return int(proc.returncode), elapsed


def _print_banner(title: str, logger: logging.Logger) -> None:
    line = "=" * 72
    logger.info(line)
    logger.info(" %s", title)
    logger.info(line)


def _print_timing_table(records: list[StepRecord], logger: logging.Logger) -> None:
    logger.info("")
    _print_banner("TIMING SUMMARY", logger)
    w = max(len(r.name) for r in records)
    sep = "-" * (w + 25)
    logger.info(sep)
    logger.info(f"{'Step':<{w}}  {'Status':>10}  {'Seconds':>10}")
    logger.info(sep)
    total = 0.0
    for r in records:
        if r.skipped:
            logger.info(f"{r.name:<{w}}  {'skipped':>10}  {'—':>10}")
            continue
        if r.returncode is None:
            logger.info(f"{r.name:<{w}}  {'not run':>10}  {'—':>10}")
            continue
        if r.returncode == 0:
            st = "ok"
            total += r.seconds
        else:
            rc = r.returncode if r.returncode is not None else "?"
            st = f"fail({rc})"
            total += r.seconds
        logger.info(f"{r.name:<{w}}  {st:>10}  {r.seconds:>10.2f}")
    logger.info(sep)
    logger.info(f"{'Total (ran steps)':<{w}}  {'':>10}  {total:>10.2f}")
    logger.info(sep)


def _awq_available() -> bool:
    try:
        importlib.import_module("awq")
        return True
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run full quantization + benchmark + plotting pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--skip-quantization", action="store_true", help="Skip steps 1–3.")
    p.add_argument("--skip-benchmarks", action="store_true", help="Skip steps 4–6.")
    p.add_argument("--skip-graphs", action="store_true", help="Skip step 7.")
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run later steps even if a previous step failed (non-zero exit).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands only; do not run.")
    p.add_argument(
        "--verify-env",
        action="store_true",
        help="Run scripts/verify_environment.py first; abort if it fails.",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose orchestrator logging.")
    p.add_argument(
        "--calibration-samples",
        type=int,
        default=None,
        help="Passed as --calibration-samples to both GPTQ scripts.",
    )
    p.add_argument(
        "--awq-max-calib-samples",
        type=int,
        default=None,
        help="Passed as --max-calib-samples to the AWQ script.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger = logging.getLogger("run_all")
    _configure_logging(args.verbose)

    root = _repo_root()
    os.chdir(root)
    logger.info("Repository root: %s", root)
    logger.info("Python: %s", sys.executable)

    _ensure_directories(root, logger)
    awq_ok = _awq_available()
    if not awq_ok:
        logger.warning("AWQ package not available; AWQ quantization step will be skipped.")

    gptq_extra: list[str] = []
    if args.calibration_samples is not None:
        gptq_extra.extend(["--calibration-samples", str(args.calibration_samples)])

    awq_extra: list[str] = []
    if args.awq_max_calib_samples is not None:
        awq_extra.extend(["--max-calib-samples", str(args.awq_max_calib_samples)])

    # (display name, script path, extra argv, skip?)
    plan: list[tuple[str, str, list[str], bool]] = []

    if args.verify_env:
        plan.append(("0. Verify environment", "scripts/verify_environment.py", [], False))

    if args.skip_quantization:
        plan.append(("1. GPTQ 4-bit", "scripts/quantize_gptq_4bit.py", gptq_extra, True))
        plan.append(("2. GPTQ 3-bit", "scripts/quantize_gptq_3bit.py", gptq_extra, True))
        plan.append(("3. AWQ 4-bit", "scripts/quantize_awq_4bit.py", awq_extra, True))
    else:
        plan.append(("1. GPTQ 4-bit", "scripts/quantize_gptq_4bit.py", gptq_extra, False))
        plan.append(("2. GPTQ 3-bit", "scripts/quantize_gptq_3bit.py", gptq_extra, False))
        plan.append(("3. AWQ 4-bit", "scripts/quantize_awq_4bit.py", awq_extra, not awq_ok))

    if args.skip_benchmarks:
        plan.append(("4. Perplexity (WikiText-2)", "scripts/evaluate_perplexity.py", [], True))
        plan.append(("5. Speed benchmark", "scripts/benchmark_speed.py", [], True))
        plan.append(("6. Memory benchmark", "scripts/benchmark_memory.py", [], True))
    else:
        plan.append(("4. Perplexity (WikiText-2)", "scripts/evaluate_perplexity.py", [], False))
        plan.append(("5. Speed benchmark", "scripts/benchmark_speed.py", [], False))
        plan.append(("6. Memory benchmark", "scripts/benchmark_memory.py", [], False))

    if args.skip_graphs:
        plan.append(("7. Generate graphs", "scripts/generate_graphs.py", [], True))
    else:
        plan.append(("7. Generate graphs", "scripts/generate_graphs.py", [], False))

    records = [
        StepRecord(name=name, script=script, skipped=skip) for name, script, extra, skip in plan
    ]
    abort_pipeline = False

    for rec, (_, script, extra, skip) in zip(records, plan):
        if skip:
            logger.info("Skipping: %s", rec.name)
            continue

        _print_banner(rec.name, logger)
        code, elapsed = _run_subprocess(root, script, extra, logger, args.dry_run)
        rec.returncode, rec.seconds = code, elapsed

        if code != 0:
            logger.error("Step failed (exit %d): %s", code, rec.name)
            if rec.name.startswith("0. Verify"):
                abort_pipeline = True
                break
            if not args.continue_on_error:
                logger.error("Stopping pipeline (use --continue-on-error to continue).")
                break
            logger.warning("Continuing after failure (--continue-on-error).")
        else:
            logger.info("Completed in %.2f s", elapsed)

    _print_timing_table(records, logger)

    if args.dry_run:
        logger.info("Dry run complete (no subprocesses executed except printed commands).")

    if abort_pipeline:
        return 1

    failed = any((not r.skipped) and (r.returncode not in (None, 0)) for r in records)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
