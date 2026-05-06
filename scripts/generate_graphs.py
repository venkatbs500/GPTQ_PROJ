#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build publication-style figures from benchmark CSVs (matplotlib only for plotting).

Reads:
    results/csv/perplexity_results.csv
    results/csv/speed_results.csv
    results/csv/memory_results.csv

Writes PNGs (300 DPI) to:
    results/figures/

Styling aims for a **NeurIPS-style** look: sans-serif type, uncluttered axes (no top/right
spines), colorblind-friendly palette, vector-friendly font settings (Type 42) if exported
to PDF later. Single-panel widths are suitable for a two-column paper (~5.5 in).

Run from repo root::

    python scripts/generate_graphs.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd


# Canonical row order for x-axis (matches benchmark script display names).
MODEL_ORDER = [
    "FP16 (original)",
    "GPTQ 4-bit",
    "GPTQ 3-bit",
    "AWQ 4-bit",
]

# Shorter tick labels (NeurIPS figures favor compact axes).
MODEL_SHORT = {
    "FP16 (original)": "FP16",
    "GPTQ 4-bit": "GPTQ 4-bit",
    "GPTQ 3-bit": "GPTQ 3-bit",
    "AWQ 4-bit": "AWQ 4-bit",
}

# Okabe–Ito palette (colorblind-safe).
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#000000"]


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


def _set_neurips_style() -> None:
    """
    Matplotlib rcParams tuned for camera-ready ML paper figures.

    - No top/right spines (Tufte-style clarity).
    - Type 42 fonts in PDF/PS (editable text in Illustrator / LaTeX workflows).
    - High savefig.dpi for raster exports; figure.dpi for on-screen preview.
    """
    mpl.rcParams.update(
        {
            "figure.figsize": (5.5, 3.2),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _read_csv(path: Path, logger: logging.Logger) -> pd.DataFrame | None:
    if not path.is_file():
        logger.warning("Missing CSV (skip): %s", path)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.exception("Failed to read %s: %s", path, exc)
        return None


def _filter_ok(df: pd.DataFrame) -> pd.DataFrame:
    if "status" not in df.columns:
        return df
    return df[df["status"].astype(str).str.lower() == "ok"].copy()


def _order_models(df: pd.DataFrame, col: str = "model") -> pd.DataFrame:
    """Stable ordering: canonical MODEL_ORDER first, then any extra rows alphabetically."""
    df = df.copy()
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    df["_sort_key"] = df[col].map(lambda n: rank.get(str(n).strip(), 10_000))
    return df.sort_values(["_sort_key", col]).drop(columns=["_sort_key"])


def _short_labels(models: list[str]) -> list[str]:
    return [MODEL_SHORT.get(m, m) for m in models]


def plot_perplexity(df: pd.DataFrame, out_path: Path, logger: logging.Logger) -> None:
    """
    Bar chart: WikiText-2 perplexity by model.

    Lower is better; we use vertical bars with value annotations. Perplexity is on a log
    scale in many papers—here we keep **linear** y by default for interpretability; the
    script could switch to ``symlog`` if dynamic ranges are huge.
    """
    df = _filter_ok(df)
    if df.empty or "perplexity" not in df.columns:
        logger.warning("No valid perplexity rows; skipping figure.")
        return
    df = _order_models(df)
    df["perplexity"] = pd.to_numeric(df["perplexity"], errors="coerce")
    df = df.dropna(subset=["perplexity"])
    if df.empty:
        logger.warning("Perplexity column empty after parse; skipping.")
        return

    models = df["model"].tolist()
    y = df["perplexity"].tolist()
    x = range(len(models))
    colors = [COLORS[i % len(COLORS)] for i in range(len(models))]

    fig, ax = plt.subplots()
    bars = ax.bar(x, y, color=colors, edgecolor="0.15", linewidth=0.4, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(_short_labels(models), rotation=18, ha="right")
    ax.set_ylabel("Perplexity (WikiText-2)")
    ax.set_xlabel("Model")
    ax.set_title("Language modeling perplexity")
    for b, val in zip(bars, y):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_path, format="png")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def plot_speed(df: pd.DataFrame, out_path: Path, logger: logging.Logger) -> None:
    """
    Bar chart: decode throughput (tokens/s) from ``benchmark_speed.py``.

    Higher is better. This summarizes the **average** greedy ``generate`` throughput
    reported in the CSV (same prompt and ``max_new_tokens`` across models).
    """
    df = _filter_ok(df)
    if df.empty or "tokens_per_sec" not in df.columns:
        logger.warning("No valid speed rows; skipping figure.")
        return
    df = _order_models(df)
    df["tokens_per_sec"] = pd.to_numeric(df["tokens_per_sec"], errors="coerce")
    df = df.dropna(subset=["tokens_per_sec"])
    if df.empty:
        logger.warning("tokens_per_sec empty after parse; skipping.")
        return

    models = df["model"].tolist()
    y = df["tokens_per_sec"].tolist()
    x = range(len(models))
    colors = [COLORS[i % len(COLORS)] for i in range(len(models))]

    fig, ax = plt.subplots()
    bars = ax.bar(x, y, color=colors, edgecolor="0.15", linewidth=0.4, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(_short_labels(models), rotation=18, ha="right")
    ax.set_ylabel("Tokens per second")
    ax.set_xlabel("Model")
    ax.set_title("Inference throughput (greedy decoding)")
    for b, val in zip(bars, y):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_path, format="png")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def plot_memory(df: pd.DataFrame, out_path: Path, logger: logging.Logger) -> None:
    """
    Grouped bar chart: peak **allocated** GPU memory (GiB).

    - **After load**: weights + buffers resident after ``from_pretrained`` / ``from_quantized``.
    - **After generate**: peak high-water mark after one ``generate()`` (KV cache / activations).

    ``memory_reserved`` is omitted here to avoid visual clutter; it is available in the CSV.
    """
    df = _filter_ok(df)
    need = {"load_peak_allocated_gib", "post_gen_peak_allocated_gib"}
    if df.empty or not need.issubset(df.columns):
        logger.warning("No valid memory rows or missing columns; skipping figure.")
        return
    df = _order_models(df)
    for c in ("load_peak_allocated_gib", "post_gen_peak_allocated_gib"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["load_peak_allocated_gib"])
    if df.empty:
        logger.warning("Memory peaks empty after parse; skipping.")
        return

    models = df["model"].tolist()
    load_p = df["load_peak_allocated_gib"].tolist()
    gen_p = df["post_gen_peak_allocated_gib"].tolist()
    has_gen = bool(df["post_gen_peak_allocated_gib"].notna().any())

    n = len(models)
    idx = range(n)
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    if has_gen:
        w = 0.36
        x0 = [i - w / 2 for i in idx]
        x1 = [i + w / 2 for i in idx]
        ax.bar(
            x0,
            load_p,
            width=w,
            label="Peak after load",
            color=COLORS[0],
            edgecolor="0.15",
            linewidth=0.4,
            zorder=3,
        )
        gen_plot = [0.0 if (g != g) else float(g) for g in gen_p]
        ax.bar(
            x1,
            gen_plot,
            width=w,
            label="Peak after 1× generate",
            color=COLORS[1],
            edgecolor="0.15",
            linewidth=0.4,
            zorder=3,
        )
        ax.legend(loc="upper left")
    else:
        ax.bar(
            list(idx),
            load_p,
            color=COLORS[0],
            edgecolor="0.15",
            linewidth=0.4,
            zorder=3,
            label="Peak after load",
        )
        ax.legend(loc="upper left")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(_short_labels(models), rotation=18, ha="right")
    ax.set_ylabel("Peak allocated (GiB)")
    ax.set_xlabel("Model")
    ax.set_title("GPU memory footprint (PyTorch allocator)")
    fig.tight_layout()
    fig.savefig(out_path, format="png")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate benchmark figures from CSVs.")
    p.add_argument("--results-dir", type=Path, default=None, help="Default: <repo>/results")
    p.add_argument("--figures-dir", type=Path, default=None, help="Default: <repo>/results/figures")
    p.add_argument("--perplexity-csv", type=Path, default=None)
    p.add_argument("--speed-csv", type=Path, default=None)
    p.add_argument("--memory-csv", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger = logging.getLogger("generate_graphs")
    _configure_logging(args.verbose)
    _set_neurips_style()

    root = _repo_root()
    res = args.results_dir or (root / "results")
    fig_dir = args.figures_dir or (res / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    ppl_csv = args.perplexity_csv or (res / "csv" / "perplexity_results.csv")
    spd_csv = args.speed_csv or (res / "csv" / "speed_results.csv")
    mem_csv = args.memory_csv or (res / "csv" / "memory_results.csv")

    df_p = _read_csv(ppl_csv, logger)
    if df_p is not None:
        plot_perplexity(df_p, fig_dir / "perplexity_comparison.png", logger)

    df_s = _read_csv(spd_csv, logger)
    if df_s is not None:
        plot_speed(df_s, fig_dir / "inference_speed.png", logger)

    df_m = _read_csv(mem_csv, logger)
    if df_m is not None:
        plot_memory(df_m, fig_dir / "memory_usage.png", logger)

    logger.info("Done. Figures directory: %s", fig_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
