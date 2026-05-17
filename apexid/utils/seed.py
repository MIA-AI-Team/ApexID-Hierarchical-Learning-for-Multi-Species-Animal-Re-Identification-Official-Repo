import os
import random

import numpy as np
import torch


DEFAULT_SEED = 42


def _seed_scipy(seed):
    try:
        import scipy
    except ImportError:
        return False

    scipy_random = getattr(scipy, "random", None)
    if scipy_random is not None and hasattr(scipy_random, "seed"):
        scipy_random.seed(seed)
    return True


def seed_everything(seed=DEFAULT_SEED, deterministic=True):
    """Seed Python, NumPy, SciPy when available, and PyTorch."""
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    _seed_scipy(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

    return seed


def seed_worker(worker_id):
    """Seed a PyTorch DataLoader worker from PyTorch's worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    _seed_scipy(worker_seed)


def make_torch_generator(seed=DEFAULT_SEED, device="cpu"):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator
