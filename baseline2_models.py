# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#%%

import numpy as np
from pathlib import Path
import heapq

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

from nkfb_util import logsumexp, cuda_if_available, torch_load

import codraw_data
from codraw_data import AbstractScene, Clipart
import abs_render
from abs_metric import scene_similarity, clipart_similarity
from episode import Episode, respond_to, response_partial

from datagen import BOWAddUpdateData, NearestNeighborData
from model import Model, select_clipart_to_tell, drawer_observe_canvas, make_fns, eval_fns
from model import scripted_tell, scripted_tell_before_peek, scripted_tell_after_peek

# %%

class BaseAddOnlyDrawer(Model, torch.nn.Module):
    datagen_cls = BOWAddUpdateData
    def init_full(self, d_hidden):
        # Helps overcome class imbalance (most cliparts are not drawn most of
        # the time)
        self.positive_scaling_coeff = 3.
        # Sigmoid is used to prevent drawing cliparts far off the canvas
        self.sigmoid_coeff = 2.
        # Scaling coefficient so that the sigmoid doesn't always saturate
        self.vals_coeff = 1. / 5.

        dg = self.datagen

        self.canvas_binary_to_hidden = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dg.NUM_BINARY, d_hidden, bias=False),
        )
        self.canvas_numerical_to_hidden = nn.Sequential(
            nn.Linear(dg.NUM_INDEX * dg.NUM_NUMERICAL, d_hidden, bias=False),
            )

        d_out = dg.NUM_INDEX * (dg.NUM_ALL + 1)
        self.hidden_to_clipart = nn.Sequential(
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def lang_to_hidden(self, msg_idxs, offsets=None):
        # Offsets is None only when batch_size is 1
        raise NotImplementedError("Subclasses should override this")

    def forward(self, example_batch):
        dg = self.datagen

        hidden_feats = (
            self.lang_to_hidden(example_batch['msg_idxs'], example_batch['offsets'])
            + self.canvas_binary_to_hidden(example_batch['canvas_binary'].float())
            + self.canvas_numerical_to_hidden(example_batch['canvas_numerical'])
            )

        clipart_scores = self.hidden_to_clipart(hidden_feats).view(-1, dg.NUM_INDEX, dg.NUM_ALL + 1)

        correct_categorical = example_batch['clipart_categorical']
        correct_numerical = example_batch['clipart_numerical']
        correct_mask = example_batch['clipart_added_mask']

        clipart_idx_scores = clipart_scores[:,:,0]
        idx_losses = F.binary_cross_entropy_with_logits(clipart_idx_scores, correct_mask.to(torch.float), reduce=False)
        idx_losses = torch.where(correct_mask, self.positive_scaling_coeff * idx_losses, idx_losses)
        per_example_idx_loss = idx_losses.sum(1)

        flat_scores = clipart_scores[:,:,1:].view((-1, dg.NUM_ALL))

        (logits_subtype, logits_depth, logits_flip, vals_numerical) = torch.split(flat_scores, [dg.NUM_SUBTYPES, dg.NUM_DEPTH, dg.NUM_FLIP, dg.NUM_NUMERICAL], dim=1)
        vals_numerical = self.sigmoid_coeff * F.sigmoid(self.vals_coeff * vals_numerical)

        subtype_losses = F.cross_entropy(logits_subtype, correct_categorical[:,:,0].view((-1,)), reduce=False).view_as(correct_categorical[:,:,0])
        depth_losses = F.cross_entropy(logits_depth, correct_categorical[:,:,1].view((-1,)), reduce=False).view_as(correct_categorical[:,:,1])
        flip_losses = F.cross_entropy(logits_flip, correct_categorical[:,:,2].view((-1,)), reduce=False).view_as(correct_categorical[:,:,2])
        vals_losses = F.mse_loss(vals_numerical, correct_numerical.view((-1, dg.NUM_NUMERICAL)), reduce=False).view_as(correct_numerical).sum(-1)
        all_losses = torch.stack([subtype_losses, depth_losses, flip_losses, vals_losses], -1).sum(-1)
        per_example_loss = torch.where(correct_mask, all_losses, all_losses.new_zeros(1)).sum(-1)

        loss = per_example_idx_loss.mean() + per_example_loss.mean()

        return loss

    @respond_to(codraw_data.ObserveCanvas)
    def draw(self, episode):
        dg = self.datagen

        msg = episode.get_last(codraw_data.TellGroup).msg
        # assert msg != ""
        words = [self.datagen.vocabulary_dict.get(word, None) for word in msg.split()]
        words = [word for word in words if word is not None]
        if not words:
            episode.append(codraw_data.DrawGroup([]))
            episode.append(codraw_data.ReplyGroup("ok"))
            return
        msg_idxs = torch.tensor(words).to(cuda_if_available)

        canvas_context = episode.get_last(codraw_data.ObserveCanvas).scene

        canvas_binary = np.zeros((dg.NUM_INDEX, 1 + dg.NUM_DEPTH + dg.NUM_FLIP), dtype=bool)
        canvas_pose = np.zeros((2, dg.NUM_SUBTYPES), dtype=bool)
        canvas_numerical = np.zeros((dg.NUM_INDEX, dg.NUM_NUMERICAL))
        for clipart in canvas_context:
            if clipart.idx in Clipart.HUMAN_IDXS:
                canvas_pose[clipart.human_idx, clipart.subtype] = True

            canvas_binary[clipart.idx, 0] = True
            canvas_binary[clipart.idx, 1 + clipart.depth] = True
            canvas_binary[clipart.idx, 1 + dg.NUM_DEPTH + clipart.flip] = True
            canvas_numerical[clipart.idx, 0] = clipart.normed_x
            canvas_numerical[clipart.idx, 1] = clipart.normed_y

        canvas_binary = np.concatenate([canvas_binary.reshape((-1,)), canvas_pose.reshape((-1,))])
        canvas_numerical = canvas_numerical.reshape((-1,))

        canvas_binary = torch.tensor(canvas_binary.astype(np.uint8), dtype=torch.uint8)[None,:].to(cuda_if_available)
        canvas_numerical = torch.tensor(canvas_numerical, dtype=torch.float)[None,:].to(cuda_if_available)

        hidden_feats = (
            self.lang_to_hidden(msg_idxs[None,:], None)
            + self.canvas_binary_to_hidden(canvas_binary.float())
            + self.canvas_numerical_to_hidden(canvas_numerical)
            )

        clipart_scores = self.hidden_to_clipart(hidden_feats).view(-1, dg.NUM_INDEX, (dg.NUM_ALL + 1))

        cliparts = []
        prior_idxs = set([c.idx for c in canvas_context])

        flat_scores = clipart_scores[:,:,1:].view((-1, dg.NUM_ALL))
        (logits_subtype, logits_depth, logits_flip, vals_numerical) = torch.split(flat_scores, [dg.NUM_SUBTYPES, dg.NUM_DEPTH, dg.NUM_FLIP, dg.NUM_NUMERICAL], dim=1)
        vals_numerical = self.sigmoid_coeff * F.sigmoid(self.vals_coeff * vals_numerical)
        vals_numerical = vals_numerical.cpu().detach().numpy()

        clipart_idx_scores = clipart_scores[0,:,0].cpu().detach().numpy()

        for idx in np.where(clipart_idx_scores > 0)[0]:
            if idx in prior_idxs:
                continue
            nx, ny = vals_numerical[idx,:]
            clipart = Clipart(idx, int(logits_subtype[idx,:].argmax()), int(logits_depth[idx,:].argmax()), int(logits_flip[idx,:].argmax()), normed_x=nx, normed_y=ny)
            cliparts.append(clipart)
        episode.append(codraw_data.DrawGroup(cliparts))
        episode.append(codraw_data.ReplyGroup("ok"))

    def get_action_fns(self):
        return [drawer_observe_canvas, self.draw]

# %%

class BOWAddOnlyDrawer(BaseAddOnlyDrawer):
    def init_full(self, d_embeddings=512, d_hidden=512):
        self._args = dict(
            d_embeddings=d_embeddings,
            d_hidden=d_hidden,
            )
        super().init_full(d_hidden)

        self.d_embeddings = d_embeddings
        self.word_embs = torch.nn.EmbeddingBag(len(self.datagen.vocabulary_dict), d_embeddings)
        self.lang_to_hidden_module = nn.Linear(d_embeddings, d_hidden)

        self.to(cuda_if_available)

    def lang_to_hidden(self, msg_idxs, offsets=None):
        bow_feats = self.word_embs(msg_idxs, offsets)
        return self.lang_to_hidden_module(bow_feats)
# %%


class LSTMAddOnlyDrawer(BaseAddOnlyDrawer):
    def init_full(self, d_embeddings=256, d_hidden=512, d_lstm=256, num_lstm_layers=1, pre_lstm_dropout=0.4, lstm_dropout=0.0):
        self._args = dict(
            d_embeddings=d_embeddings,
            d_hidden=d_hidden,
            d_lstm=256,
            num_lstm_layers=num_lstm_layers,
            pre_lstm_dropout=pre_lstm_dropout,
            lstm_dropout=lstm_dropout,
            )
        super().init_full(d_hidden)

        self.d_embeddings = d_embeddings
        self.word_embs = torch.nn.Embedding(len(self.datagen.vocabulary_dict), d_embeddings)
        self.pre_lstm_dropout = nn.Dropout(pre_lstm_dropout)
        self.lstm = nn.LSTM(d_embeddings, d_lstm, bidirectional=True, num_layers=num_lstm_layers, dropout=lstm_dropout)
        # self.post_lstm_project = nn.Linear(d_lstm * 2 * num_lstm_layers, d_hidden)
        # self.post_lstm_project = lambda x: x #nn.Linear(d_lstm * 2 * num_lstm_layers, d_hidden)
        self.post_lstm_project = lambda x: x[:,:d_hidden]
        self.to(cuda_if_available)

    def lang_to_hidden(self, msg_idxs, offsets=None):
        # global dump
        # dump = msg_idxs, offsets
        # assert False
        # bow_feats = self.word_embs(msg_idxs, offsets)
        # return self.lang_to_hidden_module(bow_feats)

        if offsets is not None:
            start = offsets.cpu()
            end = torch.cat([start[1:], torch.tensor([msg_idxs.shape[-1]])])
            undo_sorting = np.zeros(start.shape[0], dtype=int)
            undo_sorting[(start - end).numpy().argsort()] = np.arange(start.shape[0], dtype=int)
            words_packed = nn.utils.rnn.pack_sequence(sorted([msg_idxs[i:j] for i, j in list(zip(start.numpy(), end.numpy()))], key=lambda x: -x.shape[0]))
        else:
            words_packed = nn.utils.rnn.pack_sequence([msg_idxs[0,:]])
            undo_sorting = np.array([0], dtype=int)
        word_vecs = embedded = nn.utils.rnn.PackedSequence(
            self.pre_lstm_dropout(self.word_embs(words_packed.data)),
            words_packed.batch_sizes)

        _, (h_final, c_final) = self.lstm(word_vecs)

        # sentence_reps = h_final[-2:,:,:].permute(1, 2, 0).contiguous().view(undo_sorting.size, -1)
        sentence_reps = c_final[-2:,:,:].permute(1, 2, 0).contiguous().view(undo_sorting.size, -1)
        sentence_reps = self.post_lstm_project(sentence_reps)

        if offsets is not None:
            sentence_reps = sentence_reps[undo_sorting]
        return sentence_reps

# %%

class PragmaticNearestNeighborTeller(Model):
    datagen_cls = NearestNeighborData

    def init_full(self, drawer_model=None, num_candidates=10):
        self.drawer_model = drawer_model
        self.num_candidates = num_candidates

    def set_drawer_model(self, drawer_model):
        self.drawer_model = drawer_model

    def get_spec(self):
        return dict(num_candidates=self.num_candidates)

    @respond_to(codraw_data.SelectClipart)
    def tell(self, episode):
        clipart = episode.get_last(codraw_data.SelectClipart).clipart
        candidate_cliparts = heapq.nlargest(self.num_candidates, self.datagen.clipart_to_msg, key=lambda cand_clipart: clipart_similarity(cand_clipart, clipart))
        # global dump
        # dump = candidate_cliparts, episode
        # assert False

        candidate_msgs = [self.datagen.clipart_to_msg[cand_clipart] for cand_clipart in candidate_cliparts]

        expected_context = [event.clipart for event in episode if isinstance(event, codraw_data.SelectClipart)][:-1]

        candidate_responses = [self.drawer_model.just_draw(msg, expected_context) for msg in candidate_msgs]

        best_idx = np.argmax([scene_similarity(response_scene, [clipart]) for response_scene in candidate_responses])

        best_msg = candidate_msgs[best_idx]

        episode.append(codraw_data.TellGroup(best_msg))

    def get_action_fns(self):
        return [select_clipart_to_tell, self.tell]

# %%

def load_baseline2():
    baseline2_specs = torch_load(Path('models/lstmaddonly_may31.pt'))

    models = {}
    for k, spec in baseline2_specs.items():
        print(k)
        models[k] = globals()[spec['class']](spec=spec)

    # TODO(nikita): serialize these models to disk
    data_nn_a = NearestNeighborData('a')
    data_nn_b = NearestNeighborData('b')
    print('teller_pragmaticnn_a')
    models['teller_pragmaticnn_a'] = PragmaticNearestNeighborTeller(data_nn_a, drawer_model=models['drawer_lstmaddonly_a'])
    print('teller_pragmaticnn_b')
    models['teller_pragmaticnn_b'] = PragmaticNearestNeighborTeller(data_nn_b, drawer_model=models['drawer_lstmaddonly_b'])

    return models
