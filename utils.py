"""Configuration, reproducibility, device checks and atomic local/Drive IO."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import numpy as np
import torch
import yaml


def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def load_config(path):
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = config.pop("extends", None)
    return merge(load_config(path.parent / parent), config) if parent else config


def set_seed(seed, deterministic=False):
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = not deterministic


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(obj, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sync_file(path, directory):
    if directory:
        destination = Path(directory) / Path(path).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        shutil.copy2(path, temporary)
        os.replace(temporary, destination)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def device_info():
    info = dict(torch=torch.__version__, cuda_build=torch.version.cuda,
                cuda_available=torch.cuda.is_available(), bf16_supported=False)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, memory_gib=p.total_memory / 2**30,
                    capability=list(torch.cuda.get_device_capability(0)),
                    bf16_supported=torch.cuda.is_bf16_supported())
    return info


def choose_device(name="auto"):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def autocast_context(device, precision):
    enabled = device.type == "cuda" and precision == "bf16" and torch.cuda.is_bf16_supported()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def grad_norm(module):
    if module is None:
        return 0.0
    norms = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(norms).sum().sqrt()) if norms else 0.0
