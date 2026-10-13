"""Reproducibility helpers: one call seeds every RNG the training pipeline touches
and switches PyTorch/cuDNN to deterministic kernels.

Sources of run-to-run noise found in this project:
  - dataset.py augmentation uses Python's `random` (never seeded before)
  - model head init / WeightedRandomSampler / DataLoader order use torch RNGs
  - cuDNN autotuning and non-deterministic CUDA kernels
"""
import os
import random

# must be set before the first CUDA matmul for deterministic cuBLAS
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def set_seed(seed, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)  # only affects child processes; the .bat sets it for this one
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id):
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)
