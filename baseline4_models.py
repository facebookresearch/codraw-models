# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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

from baseline3_models import SceneToSeqTeller

# %%

def process_episode(episode,
        brw_rewards, brw_discounted_rewards,
        utterance_penalty,
        gamma,
        uninformative_penalty,
        ):
    scene_sims = None
    for event in episode:
        if isinstance(event, codraw_data.ObserveTruth):
            drawn_scene = []
            true_scene = event.scene
            scene_sims = []
            reward_idxs = []
            yield event
        elif isinstance(event, codraw_data.TellGroup):
            if reward_idxs:
                base_idx = reward_idxs[-1] + 1
            else:
                base_idx = 0
            offset = len(event.msg.split())
            if offset >= 50:
                offset = 50 - 1
            reward_idxs.append(base_idx + offset)
            yield event
        elif isinstance(event, (codraw_data.ObserveCanvas, codraw_data.ReplyGroup)):
            yield event
        elif isinstance(event, codraw_data.DrawGroup):
            assert drawn_scene is not None
            drawn_scene = [c for c in drawn_scene if c.idx not in [c2.idx for c2 in event.cliparts]]
            drawn_scene.extend(event.cliparts)
            scene_sims.append(scene_similarity(drawn_scene, true_scene))
            yield codraw_data.SetDrawing(drawn_scene)
        elif isinstance(event, codraw_data.SetDrawing):
            scene_sims.append(scene_similarity(event.scene, true_scene))
            yield event

    if scene_sims is not None:
        rewards = np.array(scene_sims) - np.array([0] + scene_sims[:-1])
        rewards = np.where(rewards > 0, rewards, -uninformative_penalty)

        if len(rewards) >= 50:
            rewards = np.array(list(rewards - utterance_penalty))
        else:
            rewards = np.array(list(rewards - utterance_penalty) + [0])
            if reward_idxs:
                reward_idxs.append(reward_idxs[-1] + 1)
            else:
                reward_idxs.append(0)

        new_brw_rewards = np.zeros(reward_idxs[-1] + 1)
        new_brw_rewards[np.array(reward_idxs)] = rewards
        brw_rewards.extend(list(new_brw_rewards))
        brw_discounted_rewards.extend(list(discount_rewards(new_brw_rewards, gamma)))

def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    # https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
    r = np.asarray(r)
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r

def examples_from_episodes(episodes, dg, utterance_penalty, gamma, uninformative_penalty):
    brw_rewards = []
    brw_discounted_rewards = []
    episodes = [list(process_episode(episode,
            brw_rewards, brw_discounted_rewards,
            utterance_penalty,
            gamma,
            uninformative_penalty,
            ))
        for episode in episodes]
    example_batch = dg.tensors_from_episodes(episodes + [[codraw_data.ObserveTruth([])]])
    example_batch['brw_rewards'] = torch.tensor(brw_rewards, dtype=torch.float,  device=cuda_if_available)
    example_batch['brw_discounted_rewards'] = torch.tensor(brw_discounted_rewards, dtype=torch.float, device=cuda_if_available)
    return example_batch

# %%

def collect_episodes(fns,
        dg,
        scenes=codraw_data.get_scenes('dev'),
        batch_size=16,
        utterance_penalty=0.25,
        gamma=0.99,
        uninformative_penalty=0.3
):
    with torch.no_grad():
        episodes = []
        for scene in np.random.choice(scenes, batch_size):
            ep = Episode.run(scene, fns)
            episodes.append(ep)

        example_batch = examples_from_episodes(
            episodes,
            dg=dg,
            utterance_penalty=utterance_penalty,
            gamma=gamma,
            uninformative_penalty=uninformative_penalty,
            )
    return episodes, example_batch

# %%

class RLSceneToSeqTeller(SceneToSeqTeller):
    def disable_dropout(self):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0

    def calc_rl_loss(self, example_batch):
        dg = self.datagen

        b_clipart_tags = self.tag_embs(example_batch['b_scene_tags']).view(-1, dg.NUM_INDEX, self.d_clipart_tags)

        packer = example_batch['packer']
        ob_clipart_tags = packer.ob_from_b(b_clipart_tags)
        ob_clipart_tags = self.pre_attn_tag_dropout(ob_clipart_tags)
        ob_scene_mask = packer.ob_from_b(example_batch['b_scene_mask'])

        brw_teller_tokens_in = example_batch['brw_teller_tokens_in']

        brw_embs = self.pre_lstm_emb_dropout(self.word_embs(brw_teller_tokens_in))
        orwb_embs = packer.orwb_from_brw_pack(brw_embs)

        orwb_attended_values_prelstm = self.attn_prelstm(orwb_embs, ob_clipart_tags, ob_clipart_tags, k_mask=ob_scene_mask)
        orwb_lstm_in = nn.utils.rnn.PackedSequence(torch.cat([
            orwb_embs.data,
            orwb_attended_values_prelstm.data,
            ], -1), orwb_embs.batch_sizes)

        orwb_lstm_out, _ = self.lstm(orwb_lstm_in)
        orwb_lstm_out = nn.utils.rnn.PackedSequence(self.post_lstm_dropout(orwb_lstm_out.data), orwb_lstm_out.batch_sizes)

        orwb_attended_values = self.attn(orwb_lstm_out, ob_clipart_tags, ob_clipart_tags, k_mask=ob_scene_mask)

        brw_pre_project = torch.cat([
            packer.brw_from_orwb_unpack(orwb_lstm_out),
            packer.brw_from_orwb_unpack(orwb_attended_values),
            ], -1)

        brw_word_logits = self.word_project(brw_pre_project)
        brw_word_losses = F.cross_entropy(brw_word_logits, example_batch['brw_teller_tokens_out'], reduce=False)

        b_word_losses = nn.utils.rnn.pad_packed_sequence(packer.orwb_from_brw_pack(brw_word_losses))[0].sum(0)
        print('mean nll', float(b_word_losses.mean()))

        # Discounting occurs at every word
        # brw_discounted_rewards = example_batch['brw_discounted_rewards'][:brw_word_losses.shape[0]]
        # XXX(nikita): clipping here seems wrong. Make sure there are no more crashes!
        brw_discounted_rewards = example_batch['brw_discounted_rewards']
        # TODO(nikita): what is the right baseline?
        baseline = 0.8
        brw_discounted_rewards = brw_discounted_rewards - baseline

        brw_rl_losses = brw_word_losses * brw_discounted_rewards

        rl_loss = brw_rl_losses.mean()

        return rl_loss


# %%

def load_baseline4():
    models = {}

    rl_spec_a = torch_load('models/rl_nodict_aug2.pt')
    models['teller_rl_a'] = RLSceneToSeqTeller(spec=rl_spec_a)
    models['teller_rl_b'] = None

    models['teller_rl_a'].eval()

    return models
