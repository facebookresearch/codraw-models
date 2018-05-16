# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Provides the Packer class, which is useful for managing a hierarchy where each
batch element has a variable number of conversation rounds, and each round may
consist of a variable number of messages.
"""

#%%

import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence

# %%

class Packer:
    def __init__(self, list_brw):
        coords = []
        b_lens = []
        br_lens = []

        coords_flat = []
        b_lens_flat = []
        for b, list_rw in enumerate(list_brw):
            b_lens.append(len(list_rw))
            len_flat = 0
            for r, list_w in enumerate(list_rw):
                br_lens.append(len(list_w))
                for w, _ in enumerate(list_w):
                    coords.append([b, r, w])
                    coords_flat.append([b, len_flat + w])
                len_flat += len(list_w)
            b_lens_flat.append(len_flat)

        self.coords_brw = np.array(coords, dtype=int)
        self.b_lens = np.array(b_lens, dtype=int)
        self.br_lens = np.array(br_lens, dtype=int)

        self.coords_flat = np.array(coords_flat, dtype=int)
        self.b_lens_flat = np.array(b_lens_flat, dtype=int)

        self.coords_br, self.indices_br2brw = np.unique(self.coords_brw[:,:-1], axis=0, return_inverse=True)
        _, self.indices_b2br = np.unique(self.coords_br[:,:-1], axis=0, return_inverse=True)
        self.indices_b2brw = self.indices_b2br[self.indices_br2brw]

        self.dense_shape = np.max(self.coords_brw, 0) + 1

        # Must use stable sorts here, which is why kind='mergesort'
        self.indices_b2sb = np.argsort(-self.b_lens, kind='mergesort')
        sort_by_num_rounds = np.argsort(-self.b_lens[self.indices_b2br], kind='mergesort')
        sort_by_round = np.argsort(self.coords_br[sort_by_num_rounds][:,-1], kind='mergesort')
        self.indices_br2srb = sort_by_num_rounds[sort_by_round]

        self.indices_br2sx = np.argsort(-self.br_lens, kind='mergesort')
        sort_by_num_words = np.argsort(-self.br_lens[self.indices_br2brw], kind='mergesort')
        sort_by_word_idx = np.argsort(self.coords_brw[sort_by_num_words][:,-1], kind='mergesort')
        self.indices_brw2swx = sort_by_num_words[sort_by_word_idx]

        _, batch_sizes_srb = np.unique(self.coords_br[self.indices_br2srb][:,-1], return_counts=True)
        _, batch_sizes_swx = np.unique(self.coords_brw[self.indices_brw2swx][:,-1], return_counts=True)
        self.batch_sizes_srb = torch.tensor(batch_sizes_srb, dtype=torch.long)
        self.batch_sizes_swx = torch.tensor(batch_sizes_swx, dtype=torch.long)

        self.indices_srb2br = np.argsort(self.indices_br2srb, kind='mergesort')
        self.indices_swx2brw = np.argsort(self.indices_brw2swx, kind='mergesort')
        self.indices_sb2b = np.argsort(self.indices_b2sb, kind='mergesort')
        self.indices_sx2br = np.argsort(self.indices_br2sx, kind='mergesort')

        # For flat
        self.indices_b2ob = np.argsort(-self.b_lens_flat, kind='mergesort')
        sort_by_flat_words = np.argsort(-self.b_lens_flat[self.indices_b2brw], kind='mergesort')
        sort_by_flat_word_idx = np.argsort(self.coords_flat[sort_by_flat_words][:,-1], kind='mergesort')
        self.indices_brw2orwb = sort_by_flat_words[sort_by_flat_word_idx]

        _, batch_sizes_orwb = np.unique(self.coords_flat[self.indices_brw2orwb][:,-1], return_counts=True)
        self.batch_sizes_orwb = torch.tensor(batch_sizes_orwb, dtype=torch.long)

        self.indices_ob2b = np.argsort(self.indices_b2ob, kind='mergesort')
        self.indices_orwb2brw = np.argsort(self.indices_brw2orwb, kind='mergesort')

    def brw_from_list(self, list_brw):
        vals = []
        for list_rw in list_brw:
            for list_w in list_rw:
                vals.extend(list_w)
        assert len(vals) == self.coords_brw.shape[0]
        if torch.is_tensor(vals[0]):
            return torch.stack(vals)
        else:
            return torch.tensor(vals)

    def br_from_list(self, list_br):
        vals = []
        for list_r in list_br:
            vals.extend(list_r)
        assert len(vals) == self.coords_br.shape[0]
        if torch.is_tensor(vals[0]):
            return torch.stack(vals)
        else:
            return torch.tensor(vals)

    def br_from_b_expand(self, b_in):
        return b_in[self.indices_b2br]

    def brw_from_br_expand(self, br_in):
        return br_in[self.indices_br2brw]

    def brw_from_b_expand(self, b_in):
        return b_in[self.indices_b2brw]

    def srb_from_br_pack(self, br_in):
        return PackedSequence(
            br_in[self.indices_br2srb],
            self.batch_sizes_srb
            )

    def swx_from_brw_pack(self, brw_in):
        return PackedSequence(
            brw_in[self.indices_brw2swx],
            self.batch_sizes_swx
            )

    def br_from_srb_unpack(self, srb_in):
        return srb_in.data[self.indices_srb2br]

    def brw_from_swx_unpack(self, swx_in):
        return swx_in.data[self.indices_swx2brw]

    def br_from_sx(self, sx_in):
        return sx_in[self.indices_sx2br]

    def b_from_sb(self, sb_in):
        return sb_in[self.indices_sb2b]

    def sx_from_br(self, br_in):
        return br_in[self.indices_br2sx]

    def sb_from_b(self, b_in):
        return b_in[self.indices_b2sb]

    # For flat
    def orwb_from_brw_pack(self, brw_in):
        return PackedSequence(
            brw_in[self.indices_brw2orwb],
            self.batch_sizes_orwb
            )

    def brw_from_orwb_unpack(self, orwb_in):
        return orwb_in.data[self.indices_orwb2brw]

    def b_from_ob(self, ob_in):
        return ob_in[self.indices_ob2b]

    def ob_from_b(self, b_in):
        return b_in[self.indices_b2ob]
