# PyTorch GPU benchmark (MNIST-like)

This small project provides `pytorch-test.py`, a lightweight script to benchmark GPU training speed using either synthetic random data (recommended for raw GPU benchmarking) or MNIST.

Quick usage:

- Synthetic (fast, no downloads):

```bash
python pytorch-test.py --synthetic --batch-size 512 --num-batches 200 --epochs 3
```

- MNIST (will download dataset if not present):

```bash
python pytorch-test.py --dataset mnist --batch-size 128 --epochs 5
```

Notes:

- Synthetic mode removes IO and CPU preprocessing variability and is the fastest way to measure raw GPU throughput (images/sec).
- The script prints epoch time and images/sec and attempts to show CUDA and GPU info if available.
- Installing `torch` and `torchvision` is required for real dataset benchmarks. See `requirements.txt`.

## Lessons learned running and benchmarking SIREN

This project includes a quick MNIST-like benchmark (`pytorch-test.py`) and we also cloned and ran the SIREN repository (https://github.com/vsitzmann/siren) to test a real implicit neural representation workflow. Below are concrete notes that will help reproduce the setup and re-run experiments later.

What I did (summary)
- Cloned the SIREN repo into `./siren` and ran `experiment_scripts/train_img.py` for a short 3-epoch smoke test.
- The repository ships an `environment.yml` (conda) with older package pins (PyTorch 1.5, CUDA 10.1). Instead of recreating it exactly, I used the system Python + PyTorch (2.8.0+cu128) and installed missing Python packages with pip.
- During the run I made one small compatibility patch to `siren/utils.py` to handle newer scikit-image API.

Key commands I ran (reproducible)
1) Clone the repo into your workspace:

```bash
git clone https://github.com/vsitzmann/siren.git siren
cd siren
```

2) Install minimal required Python packages (example; adjust for your environment):

```bash
pip install matplotlib scipy scikit-image imageio scikit-video h5py opencv-python cmapy tensorboard tqdm configargparse
```

3) Run a short smoke test (3 epochs, batch_size 1):

```bash
python3 experiment_scripts/train_img.py --experiment_name test_siren_img --num_epochs 3 --batch_size 1 \
	--steps_til_summary 10000 --epochs_til_ckpt 10000
```

Notes on environment and packages
- The original `environment.yml` is pinned for conda and older Python/PyTorch (1.5). Recreating that exact environment is possible with conda but may be heavy; using the system PyTorch + pip-installed packages worked well for a quick test.
- Packages I installed via pip while iterating: matplotlib, scipy, scikit-image, imageio, scikit-video, h5py, opencv-python, cmapy, tensorboard, tqdm, configargparse. You may prefer to install the equivalent conda packages if using conda.

Compatibility patch applied
- File: `siren/utils.py`
- Problem: `skimage.measure.compare_ssim` / `compare_psnr` are deprecated or moved in newer scikit-image.
- Fix: Prefer `skimage.metrics.structural_similarity` and `skimage.metrics.peak_signal_noise_ratio` when available and fall back to the older API. This is a small, local change to keep the repo runnable with modern scikit-image.

Observed runtime characteristics (example)
- Hardware: NVIDIA RTX A5000, CUDA 12.8, PyTorch 2.8
- Short SIREN run (3 epochs): completed; training loop ran at roughly ~1.4–1.8 iterations/sec for the default per-iteration workload on this machine. Note: SIREN iteration semantics differ from typical minibatches — each iteration processes all pixel-coordinate samples (implicit representation) for an image, so iteration/sec is not directly comparable to e.g., images/sec in CNN benchmarks.

Recommendations for benchmarking SIREN-style INR workloads
- Use a warmup run and then timed runs to get steady-state numbers (the SIREN training script already writes summaries/checkpoints; you can control frequency with `--steps_til_summary` and `--epochs_til_ckpt`).
- If you want throughput numbers for forward vs backward separately, I can instrument `training.py` to time forward() and backward() sections and report mean/median/p95.
- To test FP16 performance, add AMP (torch.cuda.amp) in the training loop and compare timings.

Next steps you might want me to take
- Re-run a longer, performance-focused SIREN benchmark with instrumentation (forward/backward timing, warmup) and optional AMP.
- Create a reproducible environment file (requirements.txt or a conda environment) capturing the pip installs I used.
- Add small wrapper scripts to run the same experiment with different configs and collect timings into a CSV for comparison.

If you'd like I can add a short `siren/README.rst` or append to this README with the exact pip freeze output and a script that automates installation and a default benchmark run.
