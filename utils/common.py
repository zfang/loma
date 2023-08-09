import gc
import random
from typing import List
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int]):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def perplexity(nlls: List[torch.Tensor], nsamples: int, seqlen: int) -> torch.Tensor:
    return torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))


def empty_cuda_cache():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
