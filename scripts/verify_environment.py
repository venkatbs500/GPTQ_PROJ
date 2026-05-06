"""Verify CUDA PyTorch and core quantization dependencies (Windows-friendly)."""

from __future__ import annotations

import sys


def main() -> int:
    import torch

    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda version (build): {torch.version.cuda}")
        print(f"device: {torch.cuda.get_device_name(0)}")

    import transformers  # noqa: F401
    import datasets  # noqa: F401
    import accelerate  # noqa: F401
    import pandas  # noqa: F401
    import matplotlib  # noqa: F401

    import auto_gptq  # noqa: F401

    awq_available = True
    try:
        import awq  # noqa: F401
    except Exception:
        awq_available = False

    if awq_available:
        print("imports: transformers, datasets, accelerate, pandas, matplotlib, auto_gptq, awq — ok")
    else:
        print("imports: transformers, datasets, accelerate, pandas, matplotlib, auto_gptq — ok")
        print("awq: optional and not available (AWQ benchmarks/quantization will be skipped)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"verify_environment failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
