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

from baseline1_models import load_baseline1
from baseline2_models import load_baseline2
from baseline3_models import load_baseline3
import model
from model import make_fns, eval_fns

# %%

compontent_evaluator = model.ComponentEvaluator.get()

# %%

models_baseline1 = load_baseline1()
models_baseline2 = load_baseline2()
models_baseline3 = load_baseline3()

# %%

tellers = [
    # ('teller_nn', (models_baseline1['teller_nn_a'], models_baseline1['teller_nn_b'])),
    # ('teller_c2seq', (models_baseline1['teller_c2seq_a'], models_baseline1['teller_c2seq_b'])),
    # ('teller_pragmaticnn', (models_baseline2['teller_pragmaticnn_a'], models_baseline2['teller_pragmaticnn_b'])),
    ('teller_scene2seq', (models_baseline3['teller_scene2seq_a'], models_baseline3['teller_scene2seq_b'])),
    ('teller_scene2seq_aux', (models_baseline3['teller_scene2seq_aux_a'], models_baseline3['teller_scene2seq_aux_b'])),
    ('teller_scene2seq_aux2', (models_baseline3['teller_scene2seq_aux2_a'], models_baseline3['teller_scene2seq_aux2_b'])),
]

drawers = [
    # ('drawer_nn', (models_baseline1['drawer_nn_a'], models_baseline1['drawer_nn_b'])),
    # ('drawer_sim', (models_baseline1['drawer_sim_a'], models_baseline1['drawer_sim_b'])),
    # ('drawer_bow2c', (models_baseline1['drawer_bow2c_a'], models_baseline1['drawer_bow2c_b'])),
    # ('drawer_bow2bce', (models_baseline1['drawer_bow2bce_a'], models_baseline1['drawer_bow2bce_b'])),
    # ('drawer_bowcanvas2bce', (models_baseline1['drawer_bowcanvas2bce_a'], models_baseline1['drawer_bowcanvas2bce_b'])),
    ('drawer_lstmaddonly', (models_baseline2['drawer_lstmaddonly_a'], models_baseline2['drawer_lstmaddonly_b'])),
]

# %%
print()

human_sims = np.array([
    scene_similarity(human_scene, true_scene)
    for true_scene, human_scene in codraw_data.get_truth_and_human_scenes('dev')
    ])

print(f"Human scene similarity: mean={human_sims.mean():.6f} std={human_sims.std():.6f} median={np.median(human_sims):.6f}")

# %%
print()
print()
# %%

limit = None
print("Teller           \t Drawer           \t Scene similarity")
for splits_group in [('ab', 'ba'), ('aa', 'bb')]:
    for teller_name, teller_pair in tellers:
        for drawer_name, drawer_pair in drawers:
            for splits in splits_group:
                sims = eval_fns(make_fns(splits, teller_pair, drawer_pair), limit=limit)
                teller_caption = f"{teller_name}_{splits[0]}"
                drawer_caption = f"{drawer_name}_{splits[1]}"
                print(f"{teller_caption:17s}\t {drawer_caption:17s}\t",  sims.mean())
    print()

# %%
print()
print()
# %%

limit = None
print("Drawer evaluations against script")
print("Drawer           \t Scene similarity")
for drawer_name, drawer_pair in drawers:
    for split in ('a', 'b'):
        sims = eval_fns(make_fns(split, model.scripted_tell, drawer_pair), limit=limit)
        drawer_caption = f"{drawer_name}_{split}"
        print(f"{drawer_caption:17s}\t",  sims.mean())

# %%
print()
print()
# %%

limit = None
print("Teller           \t Drawer           \t  Dir   \t Expr(human)\t Pose(human)\t Depth  \t xy (sq.)\t x-only  \t y-only")
for splits_group in [('ab', 'ba'), ('aa', 'bb')]:
    for teller_name, teller_pair in tellers:
        for drawer_name, drawer_pair in drawers:
            for splits in splits_group:
                components = compontent_evaluator.eval_fns(make_fns(splits, teller_pair, drawer_pair), limit=limit)
                teller_caption = f"{teller_name}_{splits[0]}"
                drawer_caption = f"{drawer_name}_{splits[1]}"
                print(f"{teller_caption:17s}\t {drawer_caption:17s}\t",  "\t".join(f"{num: .6f}" for num in components))
    print()

# %%
print()
print()
# %%

limit = None
print("Drawer evaluations against script")
print("Drawer           \t  Dir   \t Expr(human)\t Pose(human)\t Depth  \t xy (sq.)\t x-only  \t y-only")
for drawer_name, drawer_pair in drawers:
    for split in ('a', 'b'):
        components = compontent_evaluator.eval_fns(make_fns(split, model.scripted_tell, drawer_pair), limit=limit)
        drawer_caption = f"{drawer_name}_{split}"
        print(f"{drawer_caption:17s}\t",  "\t".join(f"{num: .6f}" for num in components))
