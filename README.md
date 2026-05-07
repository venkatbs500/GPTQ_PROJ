# LLM quantization benchmarks (GPTQ + optional AWQ)

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%20(cu121)-76B900.svg)](https://pytorch.org/get-started/locally/)
[![Platform](https://img.shields.io/badge/platform-Windows-0078D6.svg)](https://www.microsoft.com/windows)

Benchmark **FP16**, **GPTQ 4-bit**, and **GPTQ 3-bit** on a small LLM (e.g. **TinyLlama**): WikiText-2 **perplexity**, **inference throughput / latency**, **GPU memory**, **plots**, and **CSV** summaries.  
**AWQ support is optional** on Windows due to compatibility constraints with some `transformers==4.37.2` environments; scripts now skip AWQ gracefully when unavailable.

---

## Table of contents-

- [Repository layout](#repository-layout)
- [Setup (exact installation order)](#setup-exact-installation-order)
- [CUDA verification](#cuda-verification)
- [Reproducing results](#reproducing-results)
- [Configuration](#configuration)
- [Troubleshooting (Windows GPU & packages)](#troubleshooting-windows-gpu--packages)
- [License](#license)

---

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── models/              # Local checkpoints / exported quantized weights (see models/checkpoints)
├── scripts/             # CLI entrypoints, config, environment checks
├── results/             # Tables, metrics, CSV exports from runs
├── graphs/              # Matplotlib (or other) figures for reports
├── notebooks/           # Exploratory analysis and ad-hoc plots
└── report/              # Final write-up, slides, or paper assets
```

---

## Setup (exact installation order)

Follow these steps **in order**. Skipping or reordering (especially installing `requirements.txt` before CUDA PyTorch) is the most common source of `torch.cuda.is_available() == False` on Windows.

### 0) Prerequisites

- Windows 10/11 (64-bit)
- **Python 3.10.x** only for this lockfile (recommended: [python.org](https://www.python.org/downloads/release/python-31011/) or your org’s 3.10 installer)
- NVIDIA GPU with a driver that supports **CUDA 12.1** user-mode components (see [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus))
- Git (optional)

Confirm the launcher sees 3.10:

```powershell
py -0p
py -3.10 --version
```

### 1) Clone (optional) and enter the repo

```powershell
git clone https://github.com/venkatbs500/GPTQ_PROJ.git
cd GPTQ_PROJ
```

### 2) Create and activate a virtual environment (Python 3.10)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3) Install PyTorch **with CUDA 12.1 (cu121)** — **before** `requirements.txt`

Install **torch**, **torchvision**, and **torchaudio** from PyTorch’s **cu121** wheel index (versions aligned with the 2.2.x line; adjust only if you intentionally change the stack):

```powershell
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

If these exact versions are no longer published on the index, open [PyTorch — Get Started](https://pytorch.org/get-started/locally/), select **Stable**, **Windows**, **Pip**, **Python 3.10**, **CUDA 12.1**, and use the command shown there — then ensure `torch.cuda.is_available()` is still `True` (see [CUDA verification](#cuda-verification)).

**Do not** run `pip install torch` without the cu121 index first: you may get a **CPU-only** build and quantization will not use the GPU.

### 4) Install project Python dependencies

```powershell
pip install -r requirements.txt
```

This installs **`transformers==4.37.2`**, **`auto-gptq==0.7.1`**, **datasets**, **accelerate**, **peft**, **pandas**, **matplotlib**, etc., with comments and pins chosen for **Windows + cu121** (see header comments in `requirements.txt`).

### 4.1) Optional AWQ install (only if needed)

```powershell
pip install autoawq==0.2.5
```

If AWQ is not installed, AWQ steps are skipped and FP16/GPTQ experiments still run end-to-end.

**Important**

- Do **not** install `auto-gptq[triton]` on Windows.
- If a tool suggests adding **`xformers`** or other extras, check whether they pull **Triton**; on Windows that often breaks installs.

### 5) Environment smoke test

```powershell
python scripts/verify_environment.py
```

---

## CUDA verification

Run these **after** step 3 (PyTorch cu121) or **after** the full install.

### Driver and GPU (system)

```powershell
nvidia-smi
```

You should see your GPU, driver version, and a CUDA version line the driver supports. If `nvidia-smi` is not found, install or repair the **NVIDIA display driver** / **CUDA-capable driver** first.

### PyTorch build and GPU visibility

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('torch_cuda_build', torch.version.cuda); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Expected for this project: `cuda_available True`, `torch_cuda_build` reporting **12.1** (PyTorch’s CUDA build label), and a valid **device** name.

### Short matmul sanity check on GPU

```powershell
python -c "import torch; x=torch.randn(4096,4096,device='cuda'); y=torch.randn(4096,4096,device='cuda'); torch.mm(x,y).sum().item(); print('ok')"
```

If this errors, the process cannot execute CUDA kernels (driver, wrong `torch` wheel, or GPU unavailable to this Python).

---

## Reproducing results

All commands assume the **repository root** and an **activated** `.venv`.

### Step 0 — Verify imports and CUDA

```powershell
python scripts/verify_environment.py
```

### Step 1 — Run the full pipeline (one command)

```powershell
python scripts/run_all.py --verify-env
```

This runs:
1. `quantize_gptq_4bit.py`
2. `quantize_gptq_3bit.py`
3. `quantize_awq_4bit.py` (**auto-skipped if AWQ package is unavailable**)
4. `evaluate_perplexity.py`
5. `benchmark_speed.py`
6. `benchmark_memory.py`
7. `generate_graphs.py`

If you want to benchmark existing checkpoints only (no new quantization run):

```powershell
python scripts/run_all.py --skip-quantization
```

---

## Configuration

Shared defaults live in **`scripts/config.yaml`**. Current runnable scripts are self-contained and primarily use CLI flags.

---

## Troubleshooting (Windows GPU & packages)

### `torch.cuda.is_available()` is `False`

| Cause | What to do |
|--------|------------|
| CPU-only `torch` installed | `pip uninstall torch torchvision torchaudio -y`, then reinstall with **step 3** using `--index-url https://download.pytorch.org/whl/cu121`. |
| Wrong venv | Confirm `where python` points inside `.venv\Scripts\`. |
| Driver too old | Update NVIDIA driver; rerun `nvidia-smi`. |

### `DLL load failed` / `c10.dll` / CUDA library errors when importing `torch`

- Reinstall **torch / torchvision / torchaudio** as a matching triplet from the **cu121** index (same step 3).
- Install the **Microsoft Visual C++ Redistributable** (latest x64) — required for many native CUDA wheels.
- Reboot after a driver upgrade if DLL errors persist.

### `nvidia-smi` works but PyTorch sees no GPU

- You are almost certainly on a **CPU** `torch` build — reinstall from the **cu121** index (step 3).
- Laptops: ensure the app runs on the **NVIDIA** GPU (Windows **Graphics settings** / vendor mux if applicable).

### AutoGPTQ / optional AutoAWQ install failures

| Symptom | What to do |
|---------|------------|
| Build / compile errors | Stay on **Python 3.10**; upgrade `pip`; ensure **step 3** completed first so `torch` headers/wheels match. |
| Triton resolution / install errors | Do **not** install `auto-gptq[triton]`. Remove any stray `triton` requirement you may have added. |
| `autoawq-kernels` wheel not found | AWQ is optional. You can continue with FP16 + GPTQ. If needed, use **64-bit** Python 3.10 and upgrade `pip`. |

### Version conflicts after upgrading one package

This repo pins **`transformers==4.37.2`** and **`auto-gptq==0.7.1`**. Upgrading `transformers` or `torch` in isolation may break GPTQ and optional AWQ paths. Prefer reinstalling in the **exact order** in [Setup](#setup-exact-installation-order) or recreate `.venv`.

### Out of GPU memory

- Lower batch size, sequence length, or model size in `scripts/config.yaml`.
- Close other GPU applications; check `nvidia-smi` for memory use.
