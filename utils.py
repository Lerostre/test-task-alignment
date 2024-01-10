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

# hopefully this works
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

# diversity estimation from SLiC
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


# other f-divergences for Beyond-RKL
def RKL_divergence(chosen_logps, rejected_logps, *args, **kwargs):
    log_ratio = chosen_logps - rejected_logps
    return log_ratio


def KL_divergence(chosen_logps, rejected_logps, *args, **kwargs):
    log_ratio = (chosen_logps - rejected_logps).double()
    reverse_u = torch.exp(-log_ratio)
    return -reverse_u + 1


def alpha_divergence(chosen_logps, rejected_logps, alpha=0.5, *args, **kwargs):
    if alpha == 0:
        return RKL_divergence(chosen_logps, rejected_logps)
    elif alpha == 1:
        return KL_divergence(chosen_logps, rejected_logps)
    elif alpha < 0 or alpha > 1:
        raise Exception("alpha should be in range (0, 1), or should it?")
    else:
        log_ratio = (chosen_logps - rejected_logps).double()
        minus_alpha_u = torch.exp(-alpha*log_ratio)
        return (1 - minus_alpha_u) / alpha
    
    
def JS_divergence(chosen_logps, rejected_logps, *args, **kwargs):
    log_ratio = (chosen_logps - rejected_logps).double()
    u = torch.exp(log_ratio)
    return torch.log(2*u) - torch.log(1+u)


# additional f-divergences that might be even better
def pearson_chi2(chosen_logps, rejected_logps, *args, **kwargs):
    log_ratio = (chosen_logps - rejected_logps).double()
    u = torch.exp(log_ratio)
    return 2*(u-1)


def neyman_chi2(chosen_logps, rejected_logps, *args, **kwargs):
    log_ratio = (chosen_logps - rejected_logps).double()
    inverse_squared_u = torch.exp(-2*log_ratio)
    return 1 - inverse_squared_u


def hellinger_distance(chosen_logps, rejected_logps, *args, **kwargs):
    log_ratio = (chosen_logps - rejected_logps).double()
    inverse_rooted_u = torch.exp(-log_ratio / 2)
    return 1 - inverse_rooted_u


def jeffrey_distance(chosen_logps, rejected_logps, *args, **kwargs):
    log_ratio = (chosen_logps - rejected_logps).double()
    reverse_u = torch.exp(-log_ratio)
    return chosen_logps-rejected_logps + 1 - reverse_u


def GAN_distance(chosen_logps, rejected_logps, *args, **kwargs):
    log_ratio = (chosen_logps - rejected_logps).double()
    u = torch.exp(log_ratio)
    return torch.log(u) - torch.log(u+1)


def total_variation_distance(chosen_logps, rejected_logps, *args, **kwargs):
    log_ratio = (chosen_logps - rejected_logps).double()
    u = torch.exp(log_ratio)
    # torch.take(torch.tensor([-1/2, 1/2]), (u > 1).long())
    tvd = torch.empty(u.shape, dtype=float)
    tvd[u > 1] = 1/2
    tvd[u <= 1] = -1/2
    return torch.tensor(tvd, requires_grad=True)


def cnp_chi2(*args, **kwargs):
    return (2*pearson_chi2(*args, **kwargs) + neyman_chi2(*args, **kwargs)) / 3


def chi_alpha(chosen_logps, rejected_logps, alpha=2, *args, **kwargs):
    log_ratio = (chosen_logps - rejected_logps).double()
    u = torch.exp(log_ratio)
    # torch.take(torch.tensor([-1/2, 1/2]), (u > 1).long())
    tensor = torch.empty(u.shape, dtype=float)
    tensor[u > 1] = alpha * torch.pow(u[u>1]-1, alpha-1)
    tensor[u <= 1] = -alpha * torch.pow(1-u[u<=1], alpha-1)
    return torch.tensor(tensor, requires_grad=True)