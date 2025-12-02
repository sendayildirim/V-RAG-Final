import os
import random
from typing import Optional

import numpy as np
import torch

try:
    from transformers import set_seed as hf_set_seed
except ImportError:  # pragma: no cover
    hf_set_seed = None


def set_global_seed(seed: Optional[int]) -> None:
    """
    Set random seeds for python, numpy, torch and transformers to ensure reproducible runs.

    Args:
        seed: Desired seed value. If None, no-op.
    """
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hf_set_seed is not None:
        hf_set_seed(seed)

    # Some libraries also respect PYTHONHASHSEED for deterministic hashing
    os.environ["PYTHONHASHSEED"] = str(seed)
