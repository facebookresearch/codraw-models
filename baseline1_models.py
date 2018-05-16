# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#%%

import numpy as np
from pathlib import Path
import editdistance

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

from datagen import NearestNeighborData, MessageSimilarityData, BOWtoClipartData, ClipartToSeqData, BOWplusCanvasToMultiData
from model import Model, select_clipart_to_tell, drawer_observe_canvas, make_fns, eval_fns, scripted_tell

# %%

class NearestNeighborTeller(Model):
    datagen_cls = NearestNeighborData

    @respond_to(codraw_data.SelectClipart)
    def tell(self, episode):
        clipart = episode.get_last(codraw_data.SelectClipart).clipart
        best_similarity = -1
        best_msg = ""
        for cand_clipart in self.datagen.clipart_to_msg:
            cand_sim = clipart_similarity(cand_clipart, clipart)
            if cand_sim > best_similarity:
                best_similarity = cand_sim
                best_msg = self.datagen.clipart_to_msg[cand_clipart]
        episode.append(codraw_data.TellGroup(best_msg))

    def get_action_fns(self):
        return [select_clipart_to_tell, self.tell]

#%%

class CharNeighborDrawer(Model):
    datagen_cls = NearestNeighborData

    @respond_to(codraw_data.TellGroup)
    def draw(self, episode):
        msg = episode.get_last(codraw_data.TellGroup).msg
        best_distance = float('inf')
        best_clipart = None
        for cand_msg in self.datagen.msg_to_clipart:
            cand_dist = editdistance.eval(cand_msg, msg)
            if cand_dist < best_distance:
                best_distance = cand_dist
                best_clipart = self.datagen.msg_to_clipart[cand_msg]


        episode.append(codraw_data.DrawClipart(best_clipart))
        episode.append(codraw_data.ReplyGroup("ok"))

    def get_action_fns(self):
        return [self.draw]

#%%

class BOWNeighborDrawer(Model, torch.nn.Module):
    datagen_cls = MessageSimilarityData

    def init_full(self, d_embeddings=512):
        self.d_embeddings = d_embeddings
        self.word_embs = torch.nn.EmbeddingBag(len(self.datagen.vocabulary_dict), d_embeddings)

        self.msg_vecs = []
        self.msg_vecs_cliparts = []
        self.null_clipart = None

    def post_init_from_spec(self):
        self.prepare_for_inference()

    def get_spec(self):
        return dict(d_embeddings=self.d_embeddings)

    def forward(self, example_batch):
        bow_feats = self.word_embs(example_batch['words'], example_batch['offsets']).reshape(-1,21,self.d_embeddings)
        # assert np.isfinite(bow_feats.data.numpy()).all()

        bow_feats_src = bow_feats[:,0,:]
        bow_feats_tgt = bow_feats[:,1:,:]

        similarity_scores = torch.bmm(bow_feats_tgt, bow_feats_src[:,:,None])[:,:,0]
        loss = F.cross_entropy(similarity_scores, torch.zeros(similarity_scores.shape[0], dtype=torch.long, device=cuda_if_available))
        return loss

    def vec_for_msg(self, msg):
        if msg == "":
            return None
        words = [self.datagen.vocabulary_dict.get(word, None) for word in msg.split()]
        words = [word for word in words if word is not None]
        if not words:
            return None

        return self.word_embs(torch.tensor([words], dtype=torch.long, device=self.word_embs.weight.device))[0,:].cpu().detach().numpy()

    def prepare_for_inference(self):
        self.msg_vecs = []
        self.msg_vecs_cliparts = []
        # sorting is important for deterministic serialization
        for msg in sorted(self.datagen.msg_to_clipart.keys()):
            clipart = self.datagen.msg_to_clipart[msg]
            vec = self.vec_for_msg(msg)
            if vec is not None:
                self.msg_vecs.append(vec)
                self.msg_vecs_cliparts.append(clipart)
            else:
                self.null_clipart = clipart

        if self.null_clipart is None:
            self.null_clipart = self.msg_vecs_cliparts[0]
        self.msg_vecs = np.array(self.msg_vecs).T

        self.eval()

    @respond_to(codraw_data.TellGroup)
    def draw(self, episode):
        msg = episode.get_last(codraw_data.TellGroup).msg
        vec = self.vec_for_msg(msg)
        if vec is not None:
            best_clipart = self.msg_vecs_cliparts[np.argmax(vec @ self.msg_vecs)]
        else:
            best_clipart = self.null_clipart

        episode.append(codraw_data.DrawClipart(best_clipart))
        episode.append(codraw_data.ReplyGroup("ok"))

    def get_action_fns(self):
        return [self.draw]

#%%

class BOWtoClipartDrawer(Model, torch.nn.Module):
    datagen_cls = BOWtoClipartData

    NUM_INDEX = 58

    NUM_SUBTYPES = 35
    NUM_DEPTH = 3
    NUM_FLIP = 2
    NUM_CATEGORICAL = 35 + 3 + 2
    NUM_NUMERICAL = 2 # x, y

    NUM_ALL = NUM_CATEGORICAL + NUM_NUMERICAL

    def init_full(self, d_embeddings=512, d_hidden=1024):
        self.d_embeddings = d_embeddings
        self.d_hidden = d_hidden
        self.word_embs = torch.nn.EmbeddingBag(len(self.datagen.vocabulary_dict), d_embeddings)

        # Sigmoid is used to prevent drawing cliparts far off the canvas
        self.sigmoid_coeff = 2.
        # Scaling coefficient so that the sigmoid doesn't always saturate
        self.vals_coeff = 1. / 5.

        d_out = self.NUM_INDEX * (self.NUM_ALL + 1)
        self.lang_to_clipart = nn.Sequential(
            nn.Linear(d_embeddings, d_hidden),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )
        self.to(cuda_if_available)

    def get_spec(self):
        return dict(d_embeddings=self.d_embeddings, d_hidden=self.d_hidden)

    def forward(self, example_batch):
        bow_feats = self.word_embs(example_batch['msg_idxs'], example_batch['offsets'])
        clipart_scores = self.lang_to_clipart(bow_feats).reshape(-1, self.NUM_INDEX, (self.NUM_ALL + 1))
        correct_index = example_batch['clipart_index']

        logits_index = clipart_scores[:,:,0]
        correct_scores = clipart_scores[torch.arange(correct_index.shape[0], dtype=torch.long, device=cuda_if_available), correct_index][:,1:]

        (logits_subtype, logits_depth, logits_flip, vals_numerical) = torch.split(correct_scores, [self.NUM_SUBTYPES, self.NUM_DEPTH, self.NUM_FLIP, self.NUM_NUMERICAL], dim=1)
        vals_numerical = self.sigmoid_coeff * F.sigmoid(self.vals_coeff * vals_numerical)
        correct_categorical = example_batch['clipart_categorical']
        correct_numerical = example_batch['clipart_numerical']

        loss = (  F.cross_entropy(logits_index, correct_index)
                + F.cross_entropy(logits_subtype, correct_categorical[:,0])
                + F.cross_entropy(logits_depth, correct_categorical[:,1])
                + F.cross_entropy(logits_flip, correct_categorical[:,2])
                + F.mse_loss(vals_numerical, correct_numerical)
                )
        return loss

    @respond_to(codraw_data.TellGroup)
    def draw(self, episode):
        msg = episode.get_last(codraw_data.TellGroup).msg
        words = [self.datagen.vocabulary_dict.get(word, None) for word in msg.split()]
        words = [word for word in words if word is not None]
        if not words:
            # XXX(nikita): this is using DrawGroup, while normally DrawClipart is used
            episode.append(codraw_data.DrawGroup([]))
            episode.append(codraw_data.ReplyGroup("ok"))
            return
        msg_idxs = torch.tensor(words).to(cuda_if_available)

        bow_feats = self.word_embs(msg_idxs[None,:])
        clipart_scores = self.lang_to_clipart(bow_feats).reshape(-1, self.NUM_INDEX, (self.NUM_ALL + 1))[0,:,:]
        best_idx = int(clipart_scores[:,0].argmax())

        (logits_subtype, logits_depth, logits_flip, vals_numerical) = torch.split(clipart_scores[best_idx,1:], [self.NUM_SUBTYPES, self.NUM_DEPTH, self.NUM_FLIP, self.NUM_NUMERICAL])
        vals_numerical = self.sigmoid_coeff * F.sigmoid(self.vals_coeff * vals_numerical)
        nx, ny = vals_numerical.cpu().detach().numpy()

        clipart = Clipart(best_idx, int(logits_subtype.argmax()), int(logits_depth.argmax()), int(logits_flip.argmax()), normed_x=nx, normed_y=ny)

        episode.append(codraw_data.DrawClipart(clipart))
        episode.append(codraw_data.ReplyGroup("ok"))

    def get_action_fns(self):
        return [self.draw]

#%%

class ClipartToSeqTeller(Model, torch.nn.Module):
    datagen_cls = ClipartToSeqData
    def init_full(self, d_word_emb=256, d_clipart_binary=256, d_clipart_numerical=256, d_clipart_hidden=1024, d_hidden=1024):
        self._args = dict(
            d_word_emb=d_word_emb,
            d_clipart_binary=d_clipart_binary,
            d_clipart_numerical=d_clipart_numerical,
            d_clipart_hidden=d_clipart_hidden,
            d_hidden=d_hidden)

        self.word_embs = nn.Embedding(len(self.datagen.vocabulary_dict), d_word_emb)
        self.binary_feature_embs = nn.Linear(self.datagen.NUM_BINARY, d_clipart_binary, bias=False)
        self.numerical_transform = nn.Sequential(
            nn.Linear(self.datagen.NUM_NUMERICAL, d_clipart_numerical),
            nn.ReLU(),
        )

        self.clipart_transform = nn.Sequential(
            nn.Linear(d_clipart_numerical + d_clipart_binary, d_clipart_hidden),
            nn.ReLU(),
            nn.Linear(d_clipart_hidden, d_hidden),
        )

        self.lstm = nn.LSTM(d_word_emb, d_hidden, num_layers=2)
        self.word_project = nn.Linear(d_hidden, len(self.datagen.vocabulary_dict))

        self.to(cuda_if_available)

    def get_spec(self):
        return self._args

    def forward(self, example_batch):
        binary_feats = self.binary_feature_embs(example_batch['clipart_binary'])
        numerical_feats = self.numerical_transform(example_batch['clipart_numerical'])
        clipart_feats = self.clipart_transform(torch.cat([binary_feats, numerical_feats], -1))

        msg_embedded = nn.utils.rnn.PackedSequence(self.word_embs(example_batch['msg_in'].data), example_batch['msg_in'].batch_sizes)

        initial_state = torch.stack([clipart_feats] * self.lstm.num_layers)
        lstm_out, _ = self.lstm(msg_embedded, (initial_state, initial_state))
        word_logits = self.word_project(lstm_out.data)

        per_word_losses = nn.utils.rnn.PackedSequence(F.cross_entropy(word_logits, example_batch['msg_out'].data, reduce=False), example_batch['msg_out'].batch_sizes)
        per_example_losses = nn.utils.rnn.pad_packed_sequence(per_word_losses)[0].sum(-1)
        loss = per_example_losses.mean()
        return loss

    @respond_to(codraw_data.SelectClipart)
    def tell(self, episode):
        clipart = episode.get_last(codraw_data.SelectClipart).clipart

        x = clipart.normed_x
        y = clipart.normed_y
        clipart_numerical = torch.tensor([x, y], dtype=torch.float)

        clipart_binary = torch.zeros(self.datagen.NUM_BINARY)

        for val, offset in zip([clipart.idx, clipart.subtype, clipart.depth, clipart.flip], self.datagen.BINARY_OFFSETS):
            clipart_binary[val + offset] = 1.

        binary_feats = self.binary_feature_embs(clipart_binary[None,:].to(cuda_if_available))
        numerical_feats = self.numerical_transform(clipart_numerical[None,:].to(cuda_if_available))
        clipart_feats = self.clipart_transform(torch.cat([binary_feats, numerical_feats], -1))

        token_idxs = [self.datagen.vocabulary_dict['<S>']]
        # lstm_state = (F.tanh(clipart_feats[None,:,:]), clipart_feats[None,:,:])
        lstm_state = torch.stack([clipart_feats] * self.lstm.num_layers)
        lstm_state = (lstm_state, lstm_state)

        for _ in range(200):
            token_emb = self.word_embs(torch.tensor(token_idxs[-1], dtype=torch.long).to(cuda_if_available))[None,None,:]
            lstm_out, lstm_state = self.lstm(token_emb, lstm_state)

            next_token = int(self.word_project(lstm_out[0,0,:]).argmax())
            token_idxs.append(next_token)
            if next_token == self.datagen.vocabulary_dict['</S>']:
                break

        msg = " ".join([self.datagen.vocabulary[i] for i in token_idxs[1:-1]])
        episode.append(codraw_data.TellGroup(msg))

    def get_action_fns(self):
        return [select_clipart_to_tell, self.tell]

#%%

class BOWtoMultiBCEDrawer(Model, torch.nn.Module):
    datagen_cls = BOWplusCanvasToMultiData

    def init_full(self, d_embeddings=512, d_hidden=1024):
        self._args = dict(
            d_embeddings=d_embeddings,
            d_hidden=d_hidden,
            )
        self.d_embeddings = d_embeddings
        self.word_embs = torch.nn.EmbeddingBag(len(self.datagen.vocabulary_dict), d_embeddings)

        # Sigmoid is used to prevent drawing cliparts far off the canvas
        self.sigmoid_coeff = 2.
        # Scaling coefficient so that the sigmoid doesn't always saturate
        self.vals_coeff = 1. / 5.

        dg = self.datagen
        d_out = dg.NUM_INDEX * (dg.NUM_ALL + 1)
        self.lang_to_clipart = nn.Sequential(
            nn.Linear(d_embeddings, d_hidden),
            # nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )
        self.to(cuda_if_available)

    def get_spec(self):
        return self._args

    def forward(self, example_batch):
        dg = self.datagen

        bow_feats = self.word_embs(example_batch['msg_idxs'], example_batch['offsets'])
        assert np.isfinite(bow_feats.cpu().detach().numpy()).all()

        clipart_scores = self.lang_to_clipart(bow_feats).view(-1, dg.NUM_INDEX, dg.NUM_ALL + 1)
        clipart_idx_scores = clipart_scores[:,:,0]
        idx_losses = F.binary_cross_entropy_with_logits(clipart_idx_scores, example_batch['clipart_chosen_mask'].to(torch.float), reduce=False)
        # idx_losses = torch.where(example_batch['clipart_chosen_mask'], 3. * idx_losses, idx_losses)
        per_example_idx_loss = idx_losses.sum(1)

        flat_scores = clipart_scores[:,:,1:].view((-1, dg.NUM_ALL))
        (logits_subtype, logits_depth, logits_flip, vals_numerical) = torch.split(flat_scores, [dg.NUM_SUBTYPES, dg.NUM_DEPTH, dg.NUM_FLIP, dg.NUM_NUMERICAL], dim=1)
        vals_numerical = self.sigmoid_coeff * F.sigmoid(self.vals_coeff * vals_numerical)
        correct_categorical = example_batch['clipart_categorical']
        correct_numerical = example_batch['clipart_numerical']

        subtype_losses = F.cross_entropy(logits_subtype, correct_categorical[:,:,0].view((-1,)), reduce=False).view_as(correct_categorical[:,:,0])
        depth_losses = F.cross_entropy(logits_depth, correct_categorical[:,:,1].view((-1,)), reduce=False).view_as(correct_categorical[:,:,1])
        flip_losses = F.cross_entropy(logits_flip, correct_categorical[:,:,2].view((-1,)), reduce=False).view_as(correct_categorical[:,:,2])
        vals_losses = F.mse_loss(vals_numerical, correct_numerical.view((-1, dg.NUM_NUMERICAL)), reduce=False).view_as(correct_numerical).sum(-1)
        all_losses = torch.stack([subtype_losses, depth_losses, flip_losses, vals_losses], -1).sum(-1)

        per_example_loss = torch.where(example_batch['clipart_chosen_mask'], all_losses, all_losses.new_zeros(1)).sum(-1)
        loss = per_example_idx_loss.mean() + per_example_loss.mean()

        return loss

    @respond_to(codraw_data.TellGroup)
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

        bow_feats = self.word_embs(msg_idxs[None,:])
        assert np.isfinite(bow_feats.cpu().detach().numpy()).all()

        clipart_scores = self.lang_to_clipart(bow_feats).view(-1, dg.NUM_INDEX, (dg.NUM_ALL + 1))

        flat_scores = clipart_scores[:,:,1:].view((-1, dg.NUM_ALL))
        (logits_subtype, logits_depth, logits_flip, vals_numerical) = torch.split(flat_scores, [dg.NUM_SUBTYPES, dg.NUM_DEPTH, dg.NUM_FLIP, dg.NUM_NUMERICAL], dim=1)
        vals_numerical = self.sigmoid_coeff * F.sigmoid(self.vals_coeff * vals_numerical)
        vals_numerical = vals_numerical.cpu().detach().numpy()

        clipart_idx_scores = clipart_scores[0,:,0].cpu().detach().numpy()

        cliparts = []
        for idx in np.where(clipart_idx_scores > 0)[0]:
            nx, ny = vals_numerical[idx,:]
            clipart = Clipart(idx, int(logits_subtype[idx,:].argmax()), int(logits_depth[idx,:].argmax()), int(logits_flip[idx,:].argmax()), normed_x=nx, normed_y=ny)
            cliparts.append(clipart)
        episode.append(codraw_data.DrawGroup(cliparts))
        episode.append(codraw_data.ReplyGroup("ok"))

    def get_action_fns(self):
        return [self.draw]

# %%

class BOWplusCanvasDrawer(Model, torch.nn.Module):
    datagen_cls = BOWplusCanvasToMultiData
    def init_full(self, d_embeddings=512, d_hidden=512):
        self._args = dict(
            d_embeddings=d_embeddings,
            d_hidden=d_hidden,
            )
        self.d_embeddings = d_embeddings
        self.word_embs = torch.nn.EmbeddingBag(len(self.datagen.vocabulary_dict), d_embeddings)

        # Helps overcome class imbalance (most cliparts are not drawn most of
        # the time)
        self.positive_scaling_coeff = 3.
        # Sigmoid is used to prevent drawing cliparts far off the canvas
        self.sigmoid_coeff = 2.
        # Scaling coefficient so that the sigmoid doesn't always saturate
        self.vals_coeff = 1. / 5.

        dg = self.datagen

        self.lang_to_hidden = nn.Linear(d_embeddings, d_hidden)
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
        self.to(cuda_if_available)

    def forward(self, example_batch):
        dg = self.datagen

        bow_feats = self.word_embs(example_batch['msg_idxs'], example_batch['offsets'])
        assert np.isfinite(bow_feats.cpu().detach().numpy()).all()

        hidden_feats = (
            self.lang_to_hidden(bow_feats)
            + self.canvas_binary_to_hidden(example_batch['canvas_binary'].float())
            + self.canvas_numerical_to_hidden(example_batch['canvas_numerical'])
            )

        clipart_scores = self.hidden_to_clipart(hidden_feats).view(-1, dg.NUM_INDEX, dg.NUM_ALL + 1)
        clipart_idx_scores = clipart_scores[:,:,0]
        idx_losses = F.binary_cross_entropy_with_logits(clipart_idx_scores, example_batch['clipart_chosen_mask'].to(torch.float), reduce=False)
        idx_losses = torch.where(example_batch['clipart_chosen_mask'], self.positive_scaling_coeff * idx_losses, idx_losses)
        per_example_idx_loss = idx_losses.sum(1)

        flat_scores = clipart_scores[:,:,1:].view((-1, dg.NUM_ALL))
        (logits_subtype, logits_depth, logits_flip, vals_numerical) = torch.split(flat_scores, [dg.NUM_SUBTYPES, dg.NUM_DEPTH, dg.NUM_FLIP, dg.NUM_NUMERICAL], dim=1)
        vals_numerical = self.sigmoid_coeff * F.sigmoid(self.vals_coeff * vals_numerical)
        correct_categorical = example_batch['clipart_categorical']
        correct_numerical = example_batch['clipart_numerical']

        subtype_losses = F.cross_entropy(logits_subtype, correct_categorical[:,:,0].view((-1,)), reduce=False).view_as(correct_categorical[:,:,0])
        depth_losses = F.cross_entropy(logits_depth, correct_categorical[:,:,1].view((-1,)), reduce=False).view_as(correct_categorical[:,:,1])
        flip_losses = F.cross_entropy(logits_flip, correct_categorical[:,:,2].view((-1,)), reduce=False).view_as(correct_categorical[:,:,2])
        vals_losses = F.mse_loss(vals_numerical, correct_numerical.view((-1, dg.NUM_NUMERICAL)), reduce=False).view_as(correct_numerical).sum(-1)
        all_losses = torch.stack([subtype_losses, depth_losses, flip_losses, vals_losses], -1).sum(-1)

        per_example_loss = torch.where(example_batch['clipart_chosen_mask'], all_losses, all_losses.new_zeros(1)).sum(-1)
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

        bow_feats = self.word_embs(msg_idxs[None,:])
        assert np.isfinite(bow_feats.cpu().detach().numpy()).all()

        hidden_feats = (
            self.lang_to_hidden(bow_feats)
            + self.canvas_binary_to_hidden(canvas_binary.float())
            + self.canvas_numerical_to_hidden(canvas_numerical)
            )

        clipart_scores = self.hidden_to_clipart(hidden_feats).view(-1, dg.NUM_INDEX, (dg.NUM_ALL + 1))

        flat_scores = clipart_scores[:,:,1:].view((-1, dg.NUM_ALL))
        (logits_subtype, logits_depth, logits_flip, vals_numerical) = torch.split(flat_scores, [dg.NUM_SUBTYPES, dg.NUM_DEPTH, dg.NUM_FLIP, dg.NUM_NUMERICAL], dim=1)
        vals_numerical = self.sigmoid_coeff * F.sigmoid(self.vals_coeff * vals_numerical)
        vals_numerical = vals_numerical.cpu().detach().numpy()

        clipart_idx_scores = clipart_scores[0,:,0].cpu().detach().numpy()

        cliparts = []
        prior_idxs = set([c.idx for c in canvas_context])
        for idx in np.where(clipart_idx_scores > 0)[0]:
            if idx in prior_idxs: # XXX: break ties in favor of earlier actions
                continue
            nx, ny = vals_numerical[idx,:]
            clipart = Clipart(idx, int(logits_subtype[idx,:].argmax()), int(logits_depth[idx,:].argmax()), int(logits_flip[idx,:].argmax()), normed_x=nx, normed_y=ny)
            cliparts.append(clipart)
        episode.append(codraw_data.DrawGroup(cliparts))
        episode.append(codraw_data.ReplyGroup("ok"))

    def get_action_fns(self):
        return [drawer_observe_canvas, self.draw]

#%%

def load_baseline1():
    baseline1_specs = torch_load(Path('models/baseline1_may31.pt'))

    models = {}
    for k, spec in baseline1_specs.items():
        print(k)
        models[k] = globals()[spec['class']](spec=spec)

    return models
