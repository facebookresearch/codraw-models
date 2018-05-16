# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from interactivity import INTERACTIVE, try_magic, try_cd
try_cd('~/dev/drawmodel/nkcodraw')

#%%

import numpy as np
from pathlib import Path
import editdistance

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

from nkfb_util import logsumexp, cuda_if_available

import codraw_data
from codraw_data import AbstractScene, Clipart
import abs_render
from abs_metric import scene_similarity, clipart_similarity
from episode import Episode, Transcriber, respond_to, response_partial

from saved_models import load_models, make_pairs
from eval_automatic import print_eval

# %%

models = load_models(1, 2, 3, 4)

# HACK while the model is still training
models['teller_rl_b'] = models['teller_scene2seq_aux2_b']

# %%

tellers = make_pairs(models,
    # 'teller_nn',
    # 'teller_pragmaticnn',
    # 'teller_scene2seq',
    # 'teller_scene2seq_aux',
    # 'teller_scene2seq_aux2',
    'teller_rl',
)

drawers = make_pairs(models,
    # 'drawer_nn',
    # 'drawer_sim',
    # 'drawer_bow2c',
    # 'drawer_bow2bce',
    # 'drawer_bowcanvas2bce',
    'drawer_lstmaddonly',
)

# %%

print()
print_eval(do_human=True)

# %%

print()
print()
print_eval(tellers, drawers, limit=None, do_pairwise=True)

# %%

print()
print()
print_eval(tellers, drawers, limit=None, do_script=True, do_components_pairwise=True, do_components_script=True)
