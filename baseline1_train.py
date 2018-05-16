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
from model import Model, select_clipart_to_tell, drawer_observe_canvas, make_fns, eval_fns, scripted_tell
from baseline1_models import NearestNeighborTeller, CharNeighborDrawer
from baseline1_models import BOWNeighborDrawer, BOWtoClipartDrawer, ClipartToSeqTeller
from baseline1_models import BOWtoMultiBCEDrawer, BOWplusCanvasDrawer

#%%

data_nn_a = NearestNeighborData('a')
data_nn_b = NearestNeighborData('b')

teller_nn_a = NearestNeighborTeller(data_nn_a)
teller_nn_b = NearestNeighborTeller(data_nn_b)
drawer_nn_a = CharNeighborDrawer(data_nn_a)
drawer_nn_b = CharNeighborDrawer(data_nn_b)

#%%

data_sim_a = MessageSimilarityData('a')
data_sim_b = MessageSimilarityData('b')

drawer_sim_a = BOWNeighborDrawer(data_sim_a)
drawer_sim_b = BOWNeighborDrawer(data_sim_b)

optimizer_sim_a = torch.optim.Adam(drawer_sim_a.parameters())
optimizer_sim_b = torch.optim.Adam(drawer_sim_b.parameters())

#%%

for epoch in range(500):
    drawer_sim_a.train()
    for num, ex in enumerate(drawer_sim_a.datagen.get_examples_batch()):
        optimizer_sim_a.zero_grad()
        loss = drawer_sim_a.forward(ex)
        loss.backward()
        optimizer_sim_a.step()

    print(f'Done epoch {epoch} loss {float(loss)}')
    if epoch % 25 == 0:
        drawer_sim_a.prepare_for_inference()
        for splits in ('aa', 'ba'):
            sims = eval_fns(make_fns(splits, (teller_nn_a, teller_nn_b), (drawer_sim_a, drawer_sim_b)), limit=100)
            print(splits, sims.mean())
drawer_sim_a.prepare_for_inference()

# %%

for epoch in range(500):
    drawer_sim_b.train()
    for num, ex in enumerate(drawer_sim_b.datagen.get_examples_batch()):
        optimizer_sim_b.zero_grad()
        loss = drawer_sim_b.forward(ex)
        loss.backward()
        optimizer_sim_b.step()

    print(f'Done epoch {epoch} loss {float(loss)}')
    if epoch % 25 == 0:
        drawer_sim_b.prepare_for_inference()
        for splits in ('ab', 'bb'):
            sims = eval_fns(make_fns(splits, (teller_nn_a, teller_nn_b), (drawer_sim_a, drawer_sim_b)), limit=100)
            print(splits, sims.mean())
drawer_sim_b.prepare_for_inference()

#%%

data_bow2c_a = BOWtoClipartData('a')
data_bow2c_b = BOWtoClipartData('b')

drawer_bow2c_a = BOWtoClipartDrawer(data_bow2c_a)
drawer_bow2c_b = BOWtoClipartDrawer(data_bow2c_b)

optimizer_bow2c_a = torch.optim.Adam(drawer_bow2c_a.parameters())
optimizer_bow2c_b = torch.optim.Adam(drawer_bow2c_b.parameters())

# %%

for epoch in range(20):
    drawer_bow2c_a.train()
    for num, ex in enumerate(drawer_bow2c_a.datagen.get_examples_batch()):
        optimizer_bow2c_a.zero_grad()
        loss = drawer_bow2c_a.forward(ex)
        loss.backward()
        optimizer_bow2c_a.step()

    print(f'Done epoch {epoch} loss {float(loss)}')
    if epoch % 5 == 0:
        for splits in ('aa', 'ba'):
            sims = eval_fns(make_fns(splits, (teller_nn_a, teller_nn_b), (drawer_bow2c_a, drawer_bow2c_b)), limit=100)
            print(splits, sims.mean())

#%%

for epoch in range(20):
    drawer_bow2c_b.train()
    for num, ex in enumerate(drawer_bow2c_b.datagen.get_examples_batch()):
        optimizer_bow2c_b.zero_grad()
        loss = drawer_bow2c_b.forward(ex)
        loss.backward()
        optimizer_bow2c_b.step()

    print(f'Done epoch {epoch} loss {float(loss)}')
    if epoch % 5 == 0:
        for splits in ('ab', 'bb'):
            sims = eval_fns(make_fns(splits, (teller_nn_a, teller_nn_b), (drawer_bow2c_a, drawer_bow2c_b)), limit=100)
            print(splits, sims.mean())
#%%

data_c2seq_a = ClipartToSeqData('a')
data_c2seq_b = ClipartToSeqData('b')

teller_c2seq_a = ClipartToSeqTeller(data_c2seq_a)
teller_c2seq_b = ClipartToSeqTeller(data_c2seq_b)

optimizer_c2seq_a = torch.optim.Adam(teller_c2seq_a.parameters())
optimizer_c2seq_b = torch.optim.Adam(teller_c2seq_b.parameters())

#%%

for epoch in range(80):
    teller_c2seq_a.train()
    for num, ex in enumerate(teller_c2seq_a.datagen.get_examples_batch()):
        optimizer_c2seq_a.zero_grad()
        loss = teller_c2seq_a(ex)
        loss.backward()
        optimizer_c2seq_a.step()

    print(f'Done epoch {epoch} loss {float(loss)}')
    if epoch % 5 == 0:
        for splits in ('aa', 'ab'):
            sims = eval_fns(make_fns(splits, (teller_c2seq_a, teller_c2seq_b), (drawer_bow2c_a, drawer_bow2c_b)), limit=100)
            print(splits, sims.mean())

    if epoch % 50 == 49:
        optimizer_c2seq_a.param_groups[0]['lr'] *= 0.5
        print("Learning rate reduced to", optimizer_c2seq_a.param_groups[0]['lr'])

#%%

for epoch in range(80):
    teller_c2seq_b.train()
    for num, ex in enumerate(teller_c2seq_b.datagen.get_examples_batch()):
        optimizer_c2seq_b.zero_grad()
        loss = teller_c2seq_b(ex)
        loss.backward()
        optimizer_c2seq_b.step()

    print(f'Done epoch {epoch} loss {float(loss)}')
    if epoch % 5 == 0:
        for splits in ('ba', 'bb'):
            sims = eval_fns(make_fns(splits, (teller_c2seq_a, teller_c2seq_b), (drawer_bow2c_a, drawer_bow2c_b)), limit=100)
            print(splits, sims.mean())

    if epoch % 50 == 49:
        optimizer_c2seq_b.param_groups[0]['lr'] *= 0.5
        print("Learning rate reduced to", optimizer_c2seq_b.param_groups[0]['lr'])

#%%

data_bowcanvas_a = BOWplusCanvasToMultiData('a')
data_bowcanvas_b = BOWplusCanvasToMultiData('b')

drawer_bow2bce_a = BOWtoMultiBCEDrawer(data_bowcanvas_a)
drawer_bow2bce_b = BOWtoMultiBCEDrawer(data_bowcanvas_b)

optimizer_bow2bce_a = torch.optim.Adam(drawer_bow2bce_a.parameters())
optimizer_bow2bce_b = torch.optim.Adam(drawer_bow2bce_b.parameters())

#%%

for epoch in range(5):
    drawer_bow2bce_a.train()
    for num, ex in enumerate(drawer_bow2bce_a.datagen.get_examples_batch()):
        optimizer_bow2bce_a.zero_grad()
        loss = drawer_bow2bce_a.forward(ex)
        loss.backward()
        optimizer_bow2bce_a.step()

    print(f'Done epoch {epoch} loss {float(loss)}')
    if epoch % 1 == 0:
        for split in ('a',):
            sims = eval_fns(make_fns(split, scripted_tell, (drawer_bow2bce_a, drawer_bow2bce_b)), limit=100)
            print(split, sims.mean())

#%%

for epoch in range(5):
    drawer_bow2bce_b.train()
    for num, ex in enumerate(drawer_bow2bce_b.datagen.get_examples_batch()):
        optimizer_bow2bce_b.zero_grad()
        loss = drawer_bow2bce_b.forward(ex)
        loss.backward()
        optimizer_bow2bce_b.step()

    print(f'Done epoch {epoch} loss {float(loss)}')
    if epoch % 1 == 0:
        for split in ('b',):
            sims = eval_fns(make_fns(split, scripted_tell, (drawer_bow2bce_a, drawer_bow2bce_b)), limit=100)
            print(split, sims.mean())

#%%

drawer_bowcanvas2bce_a = BOWplusCanvasDrawer(data_bowcanvas_a)
drawer_bowcanvas2bce_b = BOWplusCanvasDrawer(data_bowcanvas_b)

optimizer_bowcanvas2bce_a = torch.optim.Adam(drawer_bowcanvas2bce_a.parameters())
optimizer_bowcanvas2bce_b = torch.optim.Adam(drawer_bowcanvas2bce_b.parameters())

#%%

for epoch in range(15):
    drawer_bowcanvas2bce_a.train()
    for num, ex in enumerate(drawer_bowcanvas2bce_a.datagen.get_examples_batch()):
        optimizer_bowcanvas2bce_a.zero_grad()
        loss = drawer_bowcanvas2bce_a.forward(ex)
        loss.backward()
        optimizer_bowcanvas2bce_a.step()

    print(f'Done epoch {epoch} loss {float(loss)}')
    if epoch % 1 == 0:
        for split in ('a',):
            sims = eval_fns(make_fns(split, scripted_tell, (drawer_bowcanvas2bce_a, drawer_bowcanvas2bce_b)), limit=100)
            print(split, sims.mean())

#%%

for epoch in range(15):
    drawer_bowcanvas2bce_b.train()
    for num, ex in enumerate(drawer_bowcanvas2bce_b.datagen.get_examples_batch()):
        optimizer_bowcanvas2bce_b.zero_grad()
        loss = drawer_bowcanvas2bce_b.forward(ex)
        loss.backward()
        optimizer_bowcanvas2bce_b.step()

    print(f'Done epoch {epoch} loss {float(loss)}')
    if epoch % 1 == 0:
        for split in ('b',):
            sims = eval_fns(make_fns(split, scripted_tell, (drawer_bowcanvas2bce_a, drawer_bowcanvas2bce_b)), limit=100)
            print(split, sims.mean())

#%%

baseline1_specs = dict(
    teller_nn_a = teller_nn_a.spec,
    teller_nn_b = teller_nn_b.spec,
    drawer_nn_a = drawer_nn_a.spec,
    drawer_nn_b = drawer_nn_b.spec,

    drawer_sim_a = drawer_sim_a.spec,
    drawer_sim_b = drawer_sim_b.spec,

    drawer_bow2c_a = drawer_bow2c_a.spec,
    drawer_bow2c_b = drawer_bow2c_b.spec,

    teller_c2seq_a = teller_c2seq_a.spec,
    teller_c2seq_b = teller_c2seq_b.spec,

    drawer_bow2bce_a = drawer_bow2bce_a.spec,
    drawer_bow2bce_b = drawer_bow2bce_b.spec,

    drawer_bowcanvas2bce_a = drawer_bowcanvas2bce_a.spec,
    drawer_bowcanvas2bce_b = drawer_bowcanvas2bce_b.spec,
)

#%%

torch.save(baseline1_specs, Path('models/baseline1.pt'))
