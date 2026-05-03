#!/usr/bin/env python3
"""Run a metric entrypoint after installing deterministic seeds.

Some metric backends import heavy GPU packages at module import time. This
launcher sets process-level environment knobs first, then seeds Python, NumPy
and PyTorch before dispatching the target script with its original argv.
"""

from __future__ import annotations

import os
import random
import re
import runpy
import sys
import builtins
from contextlib import ContextDecorator
from pathlib import Path


class _NullAutocast(ContextDecorator):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def install_deterministic_autocast_patch(torch_module) -> None:
    """Disable AMP autocast before metric backends import decorated modules."""
    # Off by default: VIPE/DROID-SLAM expects some tensors to be produced under
    # AMP, and globally disabling autocast can create half/float mismatches.
    if os.environ.get("REDIRECT4D_DISABLE_AMP", "0") in {"0", "false", "False"}:
        return
    if getattr(torch_module, "_redirect4d_autocast_patched", False):
        return
    torch_module.amp.autocast = _NullAutocast
    torch_module.cuda.amp.autocast = _NullAutocast
    torch_module._redirect4d_autocast_patched = True


def install_cuda_cumsum_cpu_patch(torch_module) -> None:
    """Route CUDA cumsum through CPU for deterministic metric inference.

    PyTorch currently warns that CUDA cumsum has no deterministic
    implementation. VIPE's GroundingDINO/SAM positional encoders use cumsum
    during inference; tiny differences there can move the final camera pose by
    a small amount. This patch keeps the public evaluator deterministic without
    editing the VIPE submodule.
    """
    if os.environ.get("REDIRECT4D_DETERMINISTIC_CUDA_CUMSUM", "1") in {"0", "false", "False"}:
        return
    if getattr(torch_module, "_redirect4d_cuda_cumsum_patched", False):
        return

    original_torch_cumsum = torch_module.cumsum
    original_tensor_cumsum = torch_module.Tensor.cumsum

    def _to_original_device(out, device):
        return out.to(device=device, non_blocking=False)

    def deterministic_torch_cumsum(input, dim, *, dtype=None, out=None):
        if isinstance(input, torch_module.Tensor) and input.is_cuda:
            result = original_torch_cumsum(input.cpu(), dim, dtype=dtype)
            result = _to_original_device(result, input.device)
            if out is not None:
                out.copy_(result)
                return out
            return result
        return original_torch_cumsum(input, dim, dtype=dtype, out=out)

    def deterministic_tensor_cumsum(self, dim, dtype=None):
        if self.is_cuda:
            result = original_tensor_cumsum(self.cpu(), dim, dtype=dtype)
            return _to_original_device(result, self.device)
        return original_tensor_cumsum(self, dim, dtype=dtype)

    torch_module.cumsum = deterministic_torch_cumsum
    torch_module.Tensor.cumsum = deterministic_tensor_cumsum
    torch_module._redirect4d_cuda_cumsum_patched = True


def install_vipe_scatter_import_patch(torch_module) -> None:
    """Patch VIPE scatter reductions before DroidNet imports them.

    VIPE vendors a small torch-scatter compatible wrapper. Its own docstring
    notes that CUDA scatter reductions use atomic writes and are therefore
    nondeterministic for floating point tensors. Camera-pose evaluation imports
    `scatter_mean` directly into DroidNet, so we patch via an import hook before
    the VIPE modules are loaded.
    """
    if os.environ.get("REDIRECT4D_DETERMINISTIC_SCATTER", "1") in {"0", "false", "False"}:
        return
    if getattr(torch_module, "_redirect4d_vipe_scatter_import_patched", False):
        return

    original_import = builtins.__import__

    def _patch_scatter_module(module):
        if getattr(module, "_redirect4d_deterministic_scatter_patched", False):
            return
        if not hasattr(module, "scatter_sum") or not hasattr(module, "broadcast"):
            return
        original_scatter_sum = module.scatter_sum

        def deterministic_scatter_sum(src, index, dim=-1, out=None, dim_size=None):
            if isinstance(src, torch_module.Tensor) and src.is_cuda:
                out_cpu = out.cpu() if isinstance(out, torch_module.Tensor) else None
                result = original_scatter_sum(src.cpu(), index.cpu(), dim, out_cpu, dim_size)
                result = result.to(device=src.device, non_blocking=False)
                if out is not None:
                    out.copy_(result)
                    return out
                return result
            return original_scatter_sum(src, index, dim, out, dim_size)

        def deterministic_scatter_add(src, index, dim=-1, out=None, dim_size=None):
            return deterministic_scatter_sum(src, index, dim, out, dim_size)

        def deterministic_scatter_mean(src, index, dim=-1, out=None, dim_size=None):
            out_result = deterministic_scatter_sum(src, index, dim, out, dim_size)
            dim_size_value = out_result.size(dim)

            index_dim = dim
            if index_dim < 0:
                index_dim = index_dim + src.dim()
            if index.dim() <= index_dim:
                index_dim = index.dim() - 1

            ones = torch_module.ones(index.size(), dtype=src.dtype, device=src.device)
            count = deterministic_scatter_sum(ones, index, index_dim, None, dim_size_value)
            count[count < 1] = 1
            count = module.broadcast(count, out_result, dim)
            if out_result.is_floating_point():
                out_result.true_divide_(count)
            else:
                out_result.div_(count, rounding_mode="floor")
            return out_result

        module.scatter_sum = deterministic_scatter_sum
        module.scatter_add = deterministic_scatter_add
        module.scatter_mean = deterministic_scatter_mean
        module._redirect4d_deterministic_scatter_patched = True

    def deterministic_import(name, globals=None, locals=None, fromlist=(), level=0):
        imported = original_import(name, globals, locals, fromlist, level)
        scatter_module = sys.modules.get("vipe.ext.scatter")
        if scatter_module is not None:
            _patch_scatter_module(scatter_module)
        return imported

    builtins.__import__ = deterministic_import
    torch_module._redirect4d_vipe_scatter_import_patched = True


def install_transformers_bert_compat_patch() -> None:
    """Keep VIPE/GroundingDINO compatible with newer transformers releases."""

    try:
        from transformers import BertModel
    except Exception:
        return

    original_get_extended_attention_mask = BertModel.get_extended_attention_mask

    def get_extended_attention_mask(self, attention_mask, input_shape=None, device=None, dtype=None):
        # GroundingDINO calls the transformers 4.x signature:
        #   get_extended_attention_mask(mask, input_shape, device)
        # Newer transformers interpret the third positional argument as dtype.
        try:
            import torch

            if isinstance(device, torch.device):
                device = None
            if dtype is None and isinstance(device, torch.dtype):
                dtype = device
            if dtype is None:
                dtype = getattr(self, "dtype", None)
        except Exception:
            pass
        try:
            return original_get_extended_attention_mask(
                self,
                attention_mask,
                input_shape,
                dtype=dtype,
            )
        except TypeError:
            return original_get_extended_attention_mask(
                self,
                attention_mask,
                input_shape,
                device,
            )

    BertModel.get_extended_attention_mask = get_extended_attention_mask

    if hasattr(BertModel, "get_head_mask"):
        return

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is None:
            return [None] * num_hidden_layers
        if hasattr(self, "_convert_head_mask_to_5d"):
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
            return head_mask

        import torch

        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.to(dtype=self.dtype if hasattr(self, "dtype") else torch.float32)
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
        return head_mask

    BertModel.get_head_mask = get_head_mask


class _PreferredVipeRoot(ContextDecorator):
    """Keep metric scripts on the repo-pinned VIPE submodule.

    The legacy camera-pose scripts hard-code a local VIPE checkout. Public
    runs should instead use this repository's `third_party/vipe` submodule so
    the evaluated code version is pinned by git.
    """

    def __init__(self, vipe_root: str | None) -> None:
        self.vipe_root = str(Path(vipe_root).resolve()) if vipe_root else ""
        self._original_path = None

    def __enter__(self):
        if not self.vipe_root:
            return self
        if self.vipe_root not in sys.path:
            sys.path.insert(0, self.vipe_root)
        self._original_path = sys.path
        vipe_root = self.vipe_root

        class _GuardedPath(list):
            def insert(self, index, path):
                path_str = str(path)
                resolved = str(Path(path_str).resolve()) if path_str else path_str
                is_vipe_source = (
                    Path(path_str).name == "vipe" and (Path(path_str) / "configs").exists()
                )
                if is_vipe_source and resolved != vipe_root:
                    return None
                return super().insert(index, path)

        sys.path = _GuardedPath(sys.path)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._original_path is not None:
            sys.path = self._original_path
        return False


def run_metric_script(metric_script: str) -> None:
    vipe_root = os.environ.get("REDIRECT4D_VIPE_ROOT", "")
    metric_script_dir = os.path.dirname(os.path.abspath(metric_script))
    if metric_script_dir not in sys.path:
        sys.path.insert(0, metric_script_dir)
    sys.argv = [metric_script, *sys.argv[2:]]
    if not vipe_root:
        runpy.run_path(metric_script, run_name="__main__")
        return

    source = Path(metric_script).read_text()
    source = re.sub(
        r"(?m)^R4D_VIPE_ROOT\s*=\s*[\"'].*?[\"']",
        f"R4D_VIPE_ROOT = {str(Path(vipe_root).resolve())!r}",
        source,
    )
    globals_dict = {
        "__name__": "__main__",
        "__file__": metric_script,
        "__package__": None,
        "__cached__": None,
    }
    with _PreferredVipeRoot(vipe_root):
        exec(compile(source, metric_script, "exec"), globals_dict)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("usage: run_seeded_metric.py <metric_script.py> [args...]")
        print()
        print("Runs a metric script after setting deterministic Python/NumPy/PyTorch/CUDA knobs.")
        raise SystemExit(0 if len(sys.argv) >= 2 else 2)

    seed = int(os.environ.get("REDIRECT4D_METRIC_SEED", "0"))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if os.environ.get("REDIRECT4D_DISABLE_CUDNN", "1") not in {"0", "false", "False"}:
            torch.backends.cudnn.enabled = False
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
        install_deterministic_autocast_patch(torch)
        install_cuda_cumsum_cpu_patch(torch)
        install_vipe_scatter_import_patch(torch)
        install_transformers_bert_compat_patch()
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
        except Exception:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    metric_script = sys.argv[1]
    run_metric_script(metric_script)


if __name__ == "__main__":
    main()
