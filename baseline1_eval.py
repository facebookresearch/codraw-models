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
from episode import Episode, respond_to, response_partial

from datagen import NearestNeighborData, MessageSimilarityData, BOWtoClipartData, ClipartToSeqData, BOWplusCanvasToMultiData
from model import Model, select_clipart_to_tell, drawer_observe_canvas, make_fns, eval_fns
from model import scripted_tell, scripted_tell_before_peek, scripted_tell_after_peek, draw_nothing
from baseline1_models import load_baseline1

# %%

models = load_baseline1()



# %%

tellers = [
    ('teller_nn', (models['teller_nn_a'], models['teller_nn_b'])),
    ('teller_c2seq', (models['teller_c2seq_a'], models['teller_c2seq_b'])),
]

drawers = [
    ('drawer_nn', (models['drawer_nn_a'], models['drawer_nn_b'])),
    ('drawer_sim', (models['drawer_sim_a'], models['drawer_sim_b'])),
    ('drawer_bow2c', (models['drawer_bow2c_a'], models['drawer_bow2c_b'])),
    ('drawer_bow2bce', (models['drawer_bow2bce_a'], models['drawer_bow2bce_b'])),
    ('drawer_bowcanvas2bce', (models['drawer_bowcanvas2bce_a'], models['drawer_bowcanvas2bce_b'])),
]

# %%

limit = None
print("Drawer evaluations against script")
for drawer_name, drawer_pair in drawers:
    for split in ('a', 'b'):
        sims = eval_fns(make_fns(split, scripted_tell, drawer_pair), limit=limit)
        print(f"{drawer_name}_{split}", sims.mean())

# %%

limit = None
print("Drawer evaluations against script before peek")
for drawer_name, drawer_pair in drawers:
    for split in ('a', 'b'):
        sims = eval_fns(make_fns(split, scripted_tell_before_peek, drawer_pair), limit=limit)
        print(f"{drawer_name}_{split}", sims.mean())

# %%

limit = None
print("Drawer evaluations against script after peek")

sims = eval_fns(make_fns('', scripted_tell_after_peek, draw_nothing), limit=limit)
print("draw_nothing", sims.mean())

for drawer_name, drawer_pair in drawers:
    for split in ('a', 'b'):
        sims = eval_fns(make_fns(split, scripted_tell_after_peek, drawer_pair), limit=limit)
        print(f"{drawer_name}_{split}", sims.mean())

# %%

limit = None
print("Teller/Drawer pair evaluations")
for teller_name, teller_pair in tellers:
    for drawer_name, drawer_pair in drawers:
        for splits in ('aa', 'ab', 'ba', 'bb'):
            sims = eval_fns(make_fns(splits, teller_pair, drawer_pair), limit=limit)
            print(f"{teller_name}_{splits[0]} {drawer_name}_{splits[1]}", sims.mean())
