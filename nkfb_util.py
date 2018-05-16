# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

__all__ = ['cpu', 'cuda_if_available', 'logsumexp', 'torch_load']

import torch

# %%

cpu = torch.device('cpu')

if torch.cuda.is_available():
    cuda_if_available = torch.device('cuda')
else:
    cuda_if_available = cpu

# %%

# https://github.com/pytorch/pytorch/issues/2591
def logsumexp(x, dim=None, keepdim=False):
    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim, keepdim=True)
    x = torch.where(
        (xm == float('inf')) | (xm == float('-inf')),
        xm,
        xm + torch.log(torch.sum(torch.exp(x - xm), dim, keepdim=True)))
    return x if keepdim else x.squeeze(dim)

# %%

def torch_load(*args, **kwargs):
    if cuda_if_available == cpu:
        return torch.load(*args, map_location=lambda storage, loc: storage, **kwargs)
    else:
        return torch.load(*args, **kwargs)
