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

import model
from model import make_fns, eval_fns

from saved_models import load_models, make_pairs

# %%

def print_human(limit=None, split='dev'):
    human_sims = np.array([
        scene_similarity(human_scene, true_scene)
        for true_scene, human_scene in codraw_data.get_truth_and_human_scenes('test')[:limit]
        ])

    print(f"Human scene similarity [{split}]: mean={human_sims.mean():.2f} std={human_sims.std():.2f} median={np.median(human_sims):.2f}")

# %%

def print_pairwise(tellers, drawers, teller_splits='ab', drawer_splits='ab', limit=None, split='dev'):
    print(f"Teller           \t Drawer           \t Scene similarity [{split}]")
    for splits_group in [('ab', 'ba'), ('aa', 'bb')]:
        for teller_name, teller_pair in tellers:
            for drawer_name, drawer_pair in drawers:
                for splits in splits_group:
                    if splits[0] not in teller_splits or splits[1] not in drawer_splits:
                        continue
                    sims = eval_fns(make_fns(splits, teller_pair, drawer_pair), limit=limit, split=split)
                    teller_caption = f"{teller_name}_{splits[0]}"
                    drawer_caption = f"{drawer_name}_{splits[1]}"
                    print(f"{teller_caption:17s}\t {drawer_caption:17s}\t {sims.mean():.2f}")
        print()

# %%

def print_script(drawers, drawer_splits='ab', limit=None, split='dev'):
    print("Drawer evaluations against script")
    print(f"Drawer           \t Scene similarity [{split}]")
    for drawer_name, drawer_pair in drawers:
        for drawer_split in drawer_splits:
            sims = eval_fns(make_fns(drawer_split, model.scripted_tell, drawer_pair), limit=limit, split=split)
            drawer_caption = f"{drawer_name}_{drawer_split}"
            print(f"{drawer_caption:17s}\t {sims.mean():.2f}")

# %%

component_evaluator = model.ComponentEvaluator.get()

# %%

def print_components_pairwise(tellers, drawers, teller_splits='ab', drawer_splits='ab', limit=None, split='dev'):
    print(f"Component evaluations [{split}]")
    print("Teller           \t Drawer           \t  Dir   \t Expr(human)\t Pose(human)\t Depth  \t xy (sq.)\t x-only  \t y-only")
    for splits_group in [('ab', 'ba'), ('aa', 'bb')]:
        for teller_name, teller_pair in tellers:
            for drawer_name, drawer_pair in drawers:
                for splits in splits_group:
                    if splits[0] not in teller_splits or splits[1] not in drawer_splits:
                        continue
                    components = component_evaluator.eval_fns(make_fns(splits, teller_pair, drawer_pair), limit=limit, split=split)
                    teller_caption = f"{teller_name}_{splits[0]}"
                    drawer_caption = f"{drawer_name}_{splits[1]}"
                    print(f"{teller_caption:17s}\t {drawer_caption:17s}\t",  "\t".join(f"{num: .6f}" for num in components))
        print()

def print_components_script(drawers, drawer_splits='ab', limit=None, split='dev'):
    print(f"Drawer evaluations against script [{split}]")
    print("Drawer           \t  Dir   \t Expr(human)\t Pose(human)\t Depth  \t xy (sq.)\t x-only  \t y-only")
    for drawer_name, drawer_pair in drawers:
        for drawer_split in drawer_splits:
            components = component_evaluator.eval_fns(make_fns(drawer_split, model.scripted_tell, drawer_pair), limit=limit, split=split)
            drawer_caption = f"{drawer_name}_{drawer_split}"
            print(f"{drawer_caption:17s}\t",  "\t".join(f"{num: .6f}" for num in components))

# %%

def print_eval(
    tellers=None, drawers=None,
    teller_splits='ab', drawer_splits='ab',
    limit=None,
    split='dev',
    do_all=False,
    do_human=False,
    do_pairwise=False,
    do_script=False,
    do_components_pairwise=False,
    do_components_script=False,
):
    if do_all:
        do_human = True
        do_pairwise = True
        do_script = True
        do_components_pairwise = True
        do_components_script = True

    print()

    if do_human:
        print_human(limit=limit, split=split)
        print()
        print()

    if do_pairwise:
        print_pairwise(tellers, drawers, teller_splits=teller_splits, drawer_splits=drawer_splits, limit=limit, split=split)
        print()
        print()

    if do_script:
        print_script(drawers, drawer_splits=drawer_splits, limit=limit, split=split)
        print()
        print()

    if do_components_pairwise:
        print_components_pairwise(tellers, drawers, teller_splits=teller_splits, drawer_splits=drawer_splits, limit=limit, split=split)
        print()
        print()

    if do_components_script:
        print_components_script(drawers, drawer_splits=drawer_splits, limit=limit, split=split)
        print()
        print()

# %%

if __name__ == '__main__':
    models = load_models()

# %%
if __name__ == '__main__':
    tellers = make_pairs(models,
        'teller_nn',
        # 'teller_pragmaticnn',
        'teller_scene2seq',
        'teller_scene2seq_aux2',
        'teller_rl',
    )

    drawers_for_script = make_pairs(models,
        'drawer_nn',
        # 'drawer_bowcanvas2bce',
        'drawer_lstmaddonly',
    )

    drawers_for_pairwise = make_pairs(models,
        'drawer_lstmaddonly',
    )

    limit=None
    split='test'

    print_eval(limit=limit, split=split, do_human=True)
    print_eval(tellers, drawers_for_pairwise, teller_splits='a', drawer_splits='b', limit=limit, split=split, do_pairwise=True)
    print_eval(tellers, drawers_for_script, teller_splits='a', drawer_splits='b', limit=limit, split=split, do_script=True)

# %%
# %%
# %%
# %%
# %%
# %%
