# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Multi-headed attention implementation
"""

#%%

import numpy as np

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

from nkfb_util import logsumexp, cuda_if_available

#%%

class AttentionSeqToMasked(nn.Module):
    def __init__(self,
            d_pre_q, d_pre_k, d_pre_v,
            d_qk, d_v, num_heads,
            attn_dropout):
        super().__init__()

        self.d_qk = d_qk
        self.d_v = d_v
        self.num_heads = num_heads

        self.q_proj = nn.Linear(d_pre_q, self.num_heads * self.d_qk)
        self.k_proj = nn.Linear(d_pre_k, self.num_heads * self.d_qk)
        self.v_proj = nn.Linear(d_pre_v, self.num_heads * self.d_v)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.d_out = self.num_heads * self.d_v

    def split_heads(self, tensor):
        """
        [...dims, a, num_heads x b] -> [...dims, num_heads, a, b]
        """
        return tensor.view(*tensor.shape[:-1], self.num_heads, -1).transpose(-3, -2)

    def join_heads(self, tensor):
        """
        [...dims, num_heads, a, b] -> [...dims, a, num_heads x b]
        """
        res = tensor.transpose(-3, -2).contiguous()
        return res.view(*res.shape[:-2], -1)

    def precompute_kv(self, pre_ks, pre_vs):
        assert not self.training
        ks = self.split_heads(self.k_proj(pre_ks))
        vs = self.split_heads(self.v_proj(pre_vs))
        return ks, vs

    def forward(self, pre_qs=None, pre_ks=None, pre_vs=None, ks=None, vs=None, k_mask=None):
        if isinstance(pre_qs, nn.utils.rnn.PackedSequence):
            pre_qs, lengths = nn.utils.rnn.pad_packed_sequence(pre_qs, batch_first=True)
        else:
            lengths = None
        qs = self.split_heads(self.q_proj(pre_qs))
        if ks is None:
            ks = self.split_heads(self.k_proj(pre_ks))
        if vs is None:
            vs = self.split_heads(self.v_proj(pre_vs))

        attn_logits = torch.matmul(qs, ks.transpose(-2, -1)) / np.sqrt(self.d_qk)

        if k_mask is not None:
            # k_mask is [batch, pre_ks.shape[1]] mask signalling which values
            # are valid attention targets
            attn_logits = torch.where(
                k_mask[:, None, None, :],
                attn_logits,
                torch.full_like(attn_logits, float('-inf'))
                )
        attn_probs = F.softmax(attn_logits, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        res = self.join_heads(torch.matmul(attn_probs, vs))
        if lengths is not None:
            res = nn.utils.rnn.pack_padded_sequence(res, lengths, batch_first=True)
        return res
