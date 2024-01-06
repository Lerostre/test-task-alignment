import numpy as np
from scipy.stats import entropy

import json
import pickle
import random
import os

from tqdm.auto import tqdm, trange

import torch
from torch import nn, optim
from torch.nn import functional as F

from collections import defaultdict
import gc
import warnings
warnings.filterwarnings("ignore")


# просто способ по максимуму зафиксировать работу ячейки
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

# способ оценки diversity из задания, кажется, прекрасен, как он есть
def token_entropy(generations, tokenizer):
    stats = defaultdict(int)
    num_tokens = 0
    for example in tqdm(generations, desc="Evaluating"):
        tokens = tokenizer.encode(example)
        for t in tokens:
            if t == tokenizer.pad_token_id:
                continue
            stats[t] += 1
            num_tokens += 1
    for k in stats.keys():
        stats[k] /= num_tokens
    return entropy(list(stats.values()))


# функции дивергенции для части 2
def RKL_divergence(chosen_logps, rejected_logps, *args, **kwargs):
    log_u = chosen_logps - rejected_logps
    return log_u


def KL_divergence(chosen_logps, rejected_logps, *args, **kwargs):
    u = torch.exp((rejected_logps - chosen_logps).double())
    return -u + 1


def alpha_divergence(chosen_logps, rejected_logps, alpha=0.5, *args, **kwargs):
    if alpha == 0:
        return RKL_divergence(chosen_logps, rejected_logps)
    elif alpha == 1:
        return KL_divergence(chosen_logps, rejected_logps)
    elif alpha < 0 or alpha > 1:
        raise Exception("alpha should be in range (0, 1), or should it?")
    else:
        u = torch.exp((chosen_logps - rejected_logps).double())
        return (1 - torch.pow(u, -alpha)) / alpha
    
    
def JS_divergence(chosen_logps, rejected_logps, *args, **kwargs):
    u = torch.exp((chosen_logps - rejected_logps).double())
    return torch.log(2*u) - torch.log(1+u)


def pearson_chi2(chosen_logps, rejected_logps, *args, **kwargs):
    u = torch.exp((chosen_logps - rejected_logps).double())
    return 2*(u-1)


def neyman_chi2(chosen_logps, rejected_logps, *args, **kwargs):
    u = torch.exp((chosen_logps - rejected_logps).double())
    return 1 - 1/torch.square(u)


def hellinger_distance(chosen_logps, rejected_logps, *args, **kwargs):
    u = torch.exp((chosen_logps - rejected_logps).double())
    return 1 - 1/torch.sqrt(u)


def jeffrey_distance(chosen_logps, rejected_logps, *args, **kwargs):
    u = torch.exp((chosen_logps - rejected_logps).double())
    return torch.log(u) + 1 - 1/u


def GAN_distance(chosen_logps, rejected_logps, *args, **kwargs):
    u = torch.exp((chosen_logps - rejected_logps).double())
    return torch.log(u) - torch.log(u+1)


def total_variance_distance(chosen_logps, rejected_logps, *args, **kwargs):
    u = torch.exp((chosen_logps - rejected_logps).double())
    # torch.take(torch.tensor([-1/2, 1/2]), (u > 1).long())
    tvd = torch.empty(u.shape, dtype=float)
    tvd[u > 1] = 1/2
    tvd[u <= 1] = -1/2
    return tvd