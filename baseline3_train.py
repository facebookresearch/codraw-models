# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from interactivity import INTERACTIVE, try_magic, try_cd
try_cd('~/dev/drawmodel/nkcodraw')

#%%

assert __name__ == "__main__", "Training script should not be imported!"

#%%

import numpy as np
from pathlib import Path

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

from nkfb_util import logsumexp, cuda_if_available

import codraw_data
from codraw_data import AbstractScene, Clipart
import abs_render
from abs_metric import scene_similarity, clipart_similarity
from episode import Episode, Transcriber, respond_to

import model
from model import make_fns, eval_fns
from model import Model
from baseline2_models import load_baseline2

from datagen import SceneToSeqData
from baseline3_models import SceneToSeqTeller


# %%

# scenes_and_scripts_dev = codraw_data.get_scenes_and_scripts('dev')

# transcribe = Transcriber(
#     'baseline3_train.py' if INTERACTIVE else __file__,
#     scenes_and_scripts=scenes_and_scripts_dev[::110],
#     scenes_description="scenes_and_scripts_dev[::110]")

# %%

models_baseline2 = load_baseline2()

# %%

drawer_lstmaddonly_a = models_baseline2['drawer_lstmaddonly_a']
drawer_lstmaddonly_b = models_baseline2['drawer_lstmaddonly_b']

# %%

data_scene2seq_a = SceneToSeqData('a')
data_scene2seq_b = SceneToSeqData('b')

# %%

def train_teller(split, teller_pair, num_epochs=50, limit=100):
    splits_pair = split + 'a', split + 'b'
    if split == 'a':
        teller = teller_pair[0]
    elif split == 'b':
        teller = teller_pair[1]
    else:
        assert False

    optimizer = torch.optim.Adam(teller.parameters())

    print('perplexity-dev', model.calc_perplexity(teller))
    print('perplexity-a', model.calc_perplexity(teller, 'a'))

    print('avg-loss-dev', teller.calc_split_loss())
    print('avg-loss-a', teller.calc_split_loss('a'))

    for epoch in range(num_epochs):
        teller.train()
        for num, ex in enumerate(teller.datagen.get_examples_batch()):
            optimizer.zero_grad()
            loss = teller(ex)
            loss.backward()
            optimizer.step()

        print(f'Done epoch {epoch} loss {float(loss)}')
        if epoch % 5 == 0:
            del ex, loss # clean up memory
            print('perplexity-dev', model.calc_perplexity(teller))
            print('perplexity-a', model.calc_perplexity(teller, 'a'))
            print('avg-loss-dev', teller.calc_split_loss())
            print('avg-loss-a', teller.calc_split_loss('a'))
            for splits in splits_pair:
                sims = eval_fns(make_fns(splits, teller_pair, (drawer_lstmaddonly_a, drawer_lstmaddonly_b)), limit=limit)
                print(splits, sims.mean())

# %%

teller_scene2seq_a = SceneToSeqTeller(data_scene2seq_a, prediction_loss_scale=0)
teller_scene2seq_b = SceneToSeqTeller(data_scene2seq_b, prediction_loss_scale=0)

train_teller('a', (teller_scene2seq_a, teller_scene2seq_b))
train_teller('b', (teller_scene2seq_a, teller_scene2seq_b))

# %% scene2seq with intermediate supervision for all clipart ids

teller_scene2seq_aux_a = SceneToSeqTeller(data_scene2seq_a)
teller_scene2seq_aux_b = SceneToSeqTeller(data_scene2seq_b)

train_teller('a', (teller_scene2seq_aux_a, teller_scene2seq_aux_b))
train_teller('b', (teller_scene2seq_aux_a, teller_scene2seq_aux_b))

# %% scene2seq with intermediate supervision only for present cliparts

teller_scene2seq_aux2_a = SceneToSeqTeller(data_scene2seq_a, predict_for_full_library=False,  prediction_loss_scale=6.)
teller_scene2seq_aux2_b = SceneToSeqTeller(data_scene2seq_b, predict_for_full_library=False,  prediction_loss_scale=6.)

train_teller('a', (teller_scene2seq_aux2_a, teller_scene2seq_aux2_b), num_epochs=40)
train_teller('b', (teller_scene2seq_aux2_a, teller_scene2seq_aux2_b), num_epochs=40)

# %%

scene2seq_specs = dict(
    teller_scene2seq_a = teller_scene2seq_a.spec,
    teller_scene2seq_b = teller_scene2seq_b.spec,
    teller_scene2seq_aux_a = teller_scene2seq_aux_a.spec,
    teller_scene2seq_aux_b = teller_scene2seq_aux_b.spec,
    teller_scene2seq_aux2_a = teller_scene2seq_aux2_a.spec,
    teller_scene2seq_aux2_b = teller_scene2seq_aux2_b.spec,
)

# %%

print()
print()
print("Saving models")
torch.save(scene2seq_specs, Path('models/scene2seq.pt'))

# %%

print()

print("Final evaluation on full dev set (scene2seq)")
for splits in ('aa', 'ab', 'ba', 'bb'):
    sims = eval_fns(make_fns(splits, (teller_scene2seq_a, teller_scene2seq_b), (drawer_lstmaddonly_a, drawer_lstmaddonly_b)), limit=None)
    print(splits, sims.mean())

print("Final evaluation on full dev set (scene2seq_aux)")
for splits in ('aa', 'ab', 'ba', 'bb'):
    sims = eval_fns(make_fns(splits, (teller_scene2seq_aux_a, teller_scene2seq_aux_b), (drawer_lstmaddonly_a, drawer_lstmaddonly_b)), limit=None)
    print(splits, sims.mean())

print("Final evaluation on full dev set (scene2seq_aux2)")
for splits in ('aa', 'ab', 'ba', 'bb'):
    sims = eval_fns(make_fns(splits, (teller_scene2seq_aux2_a, teller_scene2seq_aux2_b), (drawer_lstmaddonly_a, drawer_lstmaddonly_b)), limit=None)
    print(splits, sims.mean())
