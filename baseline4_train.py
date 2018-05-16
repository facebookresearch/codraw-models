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

from nkfb_util import logsumexp, cuda_if_available, torch_load
from attention import AttentionSeqToMasked

import codraw_data
from codraw_data import AbstractScene, Clipart
import abs_render
from abs_metric import scene_similarity, clipart_similarity
from episode import Episode, Transcriber, respond_to

from model import make_fns, eval_fns
from model import Model

from baseline2_models import load_baseline2
from baseline3_models import load_baseline3
from baseline4_models import RLSceneToSeqTeller, collect_episodes

# %%

models_baseline2 = load_baseline2()
models_baseline3 = load_baseline3()

# %%

drawer_lstmaddonly_a, drawer_lstmaddonly_b = models_baseline2['drawer_lstmaddonly_a'], models_baseline2['drawer_lstmaddonly_b']

teller_scene2seq_aux2_a, teller_scene2seq_aux2_b = models_baseline3['teller_scene2seq_aux2_a'], models_baseline3['teller_scene2seq_aux2_b']

# %%

def train_teller(split, teller_pair, scenes,
        utterance_penalty=0.1,
        gamma=0.999,
        uninformative_penalty=0.3,
        batch_size=16,
        num_batches=12500,
        eval_every=2000,
        lr=0.00007,
        limit=100,
        base_name="scene2seq_rl",
):
    print("Training hyperparameters:")
    for param in ['utterance_penalty',
                    'gamma',
                    'uninformative_penalty',
                    'batch_size',
                    'num_batches',
                    'lr',
                    'limit',
                ]:
        print(param, '=', locals()[param])

    drawer_pair = drawer_lstmaddonly_a, drawer_lstmaddonly_b

    splits_pair = split + 'a', split + 'b'
    if split == 'a':
        teller = teller_pair[0]
    elif split == 'b':
        teller = teller_pair[1]
    else:
        assert False

    teller.disable_dropout()
    fns = make_fns(split + split, teller_pair, drawer_pair)
    optimizer = torch.optim.Adam(teller.parameters(), lr=lr)

    def validate():
        for inference_method in ['greedy', 'sample']:
            teller.inference_method = inference_method
            for splits in splits_pair:
                sims = eval_fns(make_fns(splits, teller_pair, drawer_pair), limit=limit)
                print(splits, f'[{inference_method}]', sims.mean())

    validate()

    teller.inference_method = 'sample'
    for batch_num in range(num_batches):
        optimizer.zero_grad()
        teller.eval()
        episodes, ex = collect_episodes(
            fns,
            teller.datagen,
            scenes=scenes,
            batch_size=batch_size,
            utterance_penalty=utterance_penalty,
            gamma=gamma,
            uninformative_penalty=uninformative_penalty,
            )

        teller.train()
        loss = teller.calc_rl_loss(ex)
        loss.backward()
        # grad_norm = nn.utils.clip_grad_norm_(teller.parameters(), float('inf'))
        # XXX(nikita): clip gradients in an attempt to stabilize. Need to see if
        # there's an underlying bug, though.
        grad_norm = nn.utils.clip_grad_norm_(teller.parameters(), 1.5)
        optimizer.step()

        mean_reward = float(ex['brw_rewards'].sum().item() / ex['b_scene_mask'].shape[0])
        mean_len = np.mean([
            len([event for event in episode if isinstance(event, codraw_data.TellGroup)])
            for episode in episodes])
        sims = np.array([episode.scene_similarity() for episode in episodes])
        mean_sim = sims.mean()
        std_sim = sims.std()
        print(f'batch {batch_num} mean-reward {mean_reward} loss {float(loss)} grad {float(grad_norm)} mean-len {mean_len} mean-sim {mean_sim} std-sim {std_sim}')

        if batch_num % 5 == 0:
            for event in episodes[-1]:
                if isinstance(event, codraw_data.TellGroup):
                    print('   >', event.msg)

        if batch_num % 50 == 0:
            del episodes, ex, loss # clean up memory
            validate()

        if batch_num > 0 and batch_num % eval_every == 0:
            teller.eval()
            print("Printing representative sampled dialogs")
            teller.inference_method = 'sample'
            episodes, ex = collect_episodes(fns, teller.datagen, scenes=scenes[:1], batch_size=5)
            for episode in episodes:
                for event in episode:
                    if isinstance(event, codraw_data.TellGroup):
                        print('   >', event.msg)
                print('similarity', episode.scene_similarity())
                print('-----')

            print("Evaluating on the full dev set")
            for inference_method in ['greedy', 'sample']:
                teller.inference_method = inference_method
                for splits in splits_pair:
                    sims = eval_fns(make_fns(splits, (teller_rl_a, teller_rl_b), (drawer_lstmaddonly_a, drawer_lstmaddonly_b)), limit=None)
                    print(splits, f'[{inference_method}]', sims.mean())

            if base_name is not None:
                print("Serializing teller to disk")
                torch.save(teller.spec, Path(f'rl_models/{base_name}_{split}_{batch_num}.pt'))

# %%

# Change this to train a different teller
TELLER_SPLIT = 'a'
# TELLER_SPLIT = 'b'

# Reduce entropy: the uncertainty in the pre-trained model isn't ideal for
# starting RL. It may be possible to adjust label smoothing in the pre-training,
# but for now just reweigh the linear layer prior to the softmax
SOFTMAX_RESCALE = 3.

# %%

teller_rl_a, teller_rl_b = None, None
if TELLER_SPLIT == 'a':
    teller_rl_a = RLSceneToSeqTeller(spec=teller_scene2seq_aux2_a.spec)
    teller_rl_a.word_project.weight.data *= SOFTMAX_RESCALE
    teller_rl_a.word_project.bias.data *= SOFTMAX_RESCALE
else:
    teller_rl_b = RLSceneToSeqTeller(spec=teller_scene2seq_aux2_b.spec)
    teller_rl_b.word_project.weight.data *= SOFTMAX_RESCALE
    teller_rl_b.word_project.bias.data *= SOFTMAX_RESCALE

# %%

print(f"Info: training on partition {TELLER_SPLIT}")
scenes = np.asarray(codraw_data.get_scenes(TELLER_SPLIT))

train_teller(
    TELLER_SPLIT,
    (teller_rl_a, teller_rl_b),
    scenes,
    utterance_penalty=0.0,
    gamma=0.995,
    uninformative_penalty=0.3,
    batch_size=16,
    num_batches=60000,
    eval_every=2000,
    lr=0.00003,
    limit=100,
    base_name="b5_utt0_lr3_clip15",
    )
