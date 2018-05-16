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

from datagen import BOWAddUpdateData
from baseline2_models import BOWAddOnlyDrawer, LSTMAddOnlyDrawer
import model
from model import make_fns, eval_fns
from model import scripted_tell, scripted_tell_before_peek, scripted_tell_after_peek

# %%

data_bowaddupdate_a = BOWAddUpdateData('a')
data_bowaddupdate_b = BOWAddUpdateData('b')

# %%

# drawer_bowaddonly_a = BOWAddOnlyDrawer(data_bowaddupdate_a)
# drawer_bowaddonly_b = BOWAddOnlyDrawer(data_bowaddupdate_b)
#
# optimizer_bowaddonly_a = torch.optim.Adam(drawer_bowaddonly_a.parameters())
# optimizer_bowaddonly_b = torch.optim.Adam(drawer_bowaddonly_b.parameters())

#%%

# for epoch in range(15):
#     drawer_bowaddonly_a.train()
#     for num, ex in enumerate(drawer_bowaddonly_a.datagen.get_examples_batch()):
#         optimizer_bowaddonly_a.zero_grad()
#         loss = drawer_bowaddonly_a.forward(ex)
#         loss.backward()
#         optimizer_bowaddonly_a.step()
#
#     print(f'Done epoch {epoch} loss {float(loss)}')
#     if epoch % 1 == 0:
#         for split in ('a',):
#             sims = eval_fns(make_fns(split, scripted_tell, (drawer_bowaddonly_a, drawer_bowaddonly_b)), limit=100)
#             print(split, sims.mean())
#
#             sims = eval_fns(make_fns(split, scripted_tell_before_peek, (drawer_bowaddonly_a, drawer_bowaddonly_b)), limit=100)
#             print(split, 'before', sims.mean())
#
#             sims = eval_fns(make_fns(split, scripted_tell_after_peek, (drawer_bowaddonly_a, drawer_bowaddonly_b)), limit=100)
#             print(split, 'after', sims.mean())
# %%

drawer_lstmaddonly_a = LSTMAddOnlyDrawer(data_bowaddupdate_a)
drawer_lstmaddonly_b = LSTMAddOnlyDrawer(data_bowaddupdate_b)

optimizer_lstmaddonly_a = torch.optim.Adam(drawer_lstmaddonly_a.parameters())
optimizer_lstmaddonly_b = torch.optim.Adam(drawer_lstmaddonly_b.parameters())

#%%

for epoch in range(15):
    drawer_lstmaddonly_a.train()
    for num, ex in enumerate(drawer_lstmaddonly_a.datagen.get_examples_batch()):
        optimizer_lstmaddonly_a.zero_grad()
        loss = drawer_lstmaddonly_a.forward(ex)
        loss.backward()
        optimizer_lstmaddonly_a.step()

    print(f'Done epoch {epoch} loss {float(loss)}')
    if epoch % 1 == 0:
        for split in ('a',):
            sims = eval_fns(make_fns(split, scripted_tell, (drawer_lstmaddonly_a, drawer_lstmaddonly_b)), limit=100)
            print(split, sims.mean())

            sims = eval_fns(make_fns(split, scripted_tell_before_peek, (drawer_lstmaddonly_a, drawer_lstmaddonly_b)), limit=100)
            print(split, 'before', sims.mean())

            sims = eval_fns(make_fns(split, scripted_tell_after_peek, (drawer_lstmaddonly_a, drawer_lstmaddonly_b)), limit=100)
            print(split, 'after', sims.mean())
#%%

for epoch in range(15):
    drawer_lstmaddonly_b.train()
    for num, ex in enumerate(drawer_lstmaddonly_b.datagen.get_examples_batch()):
        optimizer_lstmaddonly_b.zero_grad()
        loss = drawer_lstmaddonly_b.forward(ex)
        loss.backward()
        optimizer_lstmaddonly_b.step()

    print(f'Done epoch {epoch} loss {float(loss)}')
    if epoch % 1 == 0:
        for split in ('b',):
            sims = eval_fns(make_fns(split, scripted_tell, (drawer_lstmaddonly_a, drawer_lstmaddonly_b)), limit=100)
            print(split, sims.mean())

            sims = eval_fns(make_fns(split, scripted_tell_before_peek, (drawer_lstmaddonly_a, drawer_lstmaddonly_b)), limit=100)
            print(split, 'before', sims.mean())

            sims = eval_fns(make_fns(split, scripted_tell_after_peek, (drawer_lstmaddonly_a, drawer_lstmaddonly_b)), limit=100)
            print(split, 'after', sims.mean())

# %%

lstmaddonly_specs = dict(
    drawer_lstmaddonly_a = drawer_lstmaddonly_a.spec,
    drawer_lstmaddonly_b = drawer_lstmaddonly_b.spec,
)

#%%

torch.save(lstmaddonly_specs, Path('models/lstmaddonly.pt'))
