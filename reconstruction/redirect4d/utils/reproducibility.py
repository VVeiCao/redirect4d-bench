"""Reproducibility helpers for Redirect4D reconstruction."""

from __future__ import annotations

import os
import random


def configure_reproducibility(
    seed: int = 23,
    deterministic: bool = True,
    warn_only: bool = True,
) -> None:
    """Seed common RNGs and request deterministic CUDA kernels where possible.

    Some VIPE/LyRA dependencies still use CUDA operations that PyTorch marks as
    non-deterministic. Keep ``warn_only=True`` for normal runs so those operators
    are reported but do not abort the reconstruction.
    """

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except Exception:
        pass
