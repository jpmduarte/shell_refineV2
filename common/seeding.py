"""Seeding for reproducible training: same weights, batches, augmentation and GPU kernels."""
import numpy as np
import torch

NUMPY_SEED_MODULUS = 2 ** 32   # numpy seeds must fit in 32 bits; PyTorch's worker seeds are 64


def make_deterministic(seed: int):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loader_generator(seed):
    """Seeds the loader's shuffling and its workers' seeds; None leaves them random."""
    return torch.Generator().manual_seed(seed) if seed is not None else None


def seed_worker(worker_id):
    """Give each loader worker its own augmentation generator; copies of one would draw in lock-step."""
    info = torch.utils.data.get_worker_info()
    info.dataset._rng = np.random.default_rng(info.seed % NUMPY_SEED_MODULUS)
