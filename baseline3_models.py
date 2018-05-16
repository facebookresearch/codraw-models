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

from datagen import SceneToSeqData
from model import make_fns, eval_fns
from model import Model

# %%

class SceneToSeqTeller(Model, torch.nn.Module):
    datagen_cls = SceneToSeqData

    def init_full(self,
            d_word_emb=256,
            d_tag_emb=128, num_heads=4, d_qkv=128,
            pre_attn_tag_dropout=0.2, attn_dropout=0.1,
            d_lstm=1024, num_lstm_layers=1,
            pre_lstm_emb_dropout=0.5,
            pre_lstm_scene_dropout=0.15,
            lstm_dropout=0.0,
            post_lstm_dropout=0.3,
            label_smoothing=0.05,
            prediction_loss_scale=5.,
            d_clipart_state_hidden=1024,
            predict_for_full_library=True,
            ):
        self._args = dict(
            d_word_emb=d_word_emb,
            d_tag_emb=d_tag_emb, num_heads=num_heads, d_qkv=d_qkv,
            pre_attn_tag_dropout=pre_attn_tag_dropout,
            attn_dropout=attn_dropout,
            d_lstm=d_lstm, num_lstm_layers=num_lstm_layers, pre_lstm_emb_dropout=pre_lstm_emb_dropout,
            pre_lstm_scene_dropout=pre_lstm_scene_dropout,
            lstm_dropout=lstm_dropout,
            post_lstm_dropout=post_lstm_dropout,
            label_smoothing=label_smoothing,
            prediction_loss_scale=prediction_loss_scale,
            d_clipart_state_hidden=d_clipart_state_hidden,
            predict_for_full_library=predict_for_full_library,
            )
        dg = self.datagen

        self.tag_embs = nn.Embedding(dg.NUM_TAGS, d_tag_emb)
        self.d_clipart_tags = d_tag_emb * dg.NUM_TAGS_PER_INDEX

        self.pre_attn_tag_dropout = nn.Dropout(pre_attn_tag_dropout)

        self.attn_prelstm = AttentionSeqToMasked(
            d_pre_q=d_word_emb,
            d_pre_k=self.d_clipart_tags,
            d_pre_v=self.d_clipart_tags,
            d_qk=d_qkv, d_v=d_qkv,
            num_heads=num_heads,
            attn_dropout=attn_dropout)

        self.attn = AttentionSeqToMasked(
            d_pre_q=d_lstm,
            d_pre_k=self.d_clipart_tags,
            d_pre_v=self.d_clipart_tags,
            d_qk=d_qkv, d_v=d_qkv,
            num_heads=num_heads,
            attn_dropout=attn_dropout)

        self.word_embs = nn.Embedding(len(self.datagen.vocabulary_dict), d_word_emb)

        self.pre_lstm_emb_dropout = nn.Dropout(pre_lstm_emb_dropout)
        self.pre_lstm_scene_dropout = nn.Dropout(pre_lstm_scene_dropout)
        self.lstm = nn.LSTM(d_word_emb + self.attn_prelstm.d_out, d_lstm, num_layers=num_lstm_layers, dropout=lstm_dropout)
        self.post_lstm_dropout = nn.Dropout(post_lstm_dropout)
        self.word_project = nn.Linear(d_lstm + self.attn.d_out, len(self.datagen.vocabulary_dict))

        self.label_smoothing = label_smoothing

        # Possible auxiliary loss for predicting clipart state
        self.prediction_loss_scale = prediction_loss_scale
        self.predict_for_full_library = predict_for_full_library
        if prediction_loss_scale > 0:
            if predict_for_full_library:
                d_clipart_state_in = d_lstm + dg.NUM_INDEX
            else:
                d_clipart_state_in = d_lstm
            self.clipart_state_predictor = nn.Sequential(
                nn.Linear(d_clipart_state_in, d_clipart_state_hidden),
                nn.ReLU(),
                nn.Linear(d_clipart_state_hidden, dg.NUM_INDEX * dg.NUM_CLIPART_STATES),
            )
        else:
            self.clipart_state_predictor = None

        self.to(cuda_if_available)

        self.inference_method = 'greedy'
        self.sampling_temperature = 1.0
        self.max_rounds = 50 # This is only changed for human eval

    def get_spec(self):
        return self._args

    def print_hparams(self):
        print("Hyperparameters:")
        for k, v in self._args.items():
            print(k, '=', v)
        print()

    def forward(self, example_batch, return_loss=True, return_nll_count=False):
        dg = self.datagen

        b_clipart_tags = self.tag_embs(example_batch['b_scene_tags']).view(-1, dg.NUM_INDEX, self.d_clipart_tags)

        if not (return_loss or return_nll_count):
            ks_prelstm, vs_prelstm = self.attn_prelstm.precompute_kv(b_clipart_tags, b_clipart_tags)
            ks, vs = self.attn.precompute_kv(b_clipart_tags, b_clipart_tags)
            return example_batch['b_scene_mask'], ks_prelstm, vs_prelstm, ks, vs

        packer = example_batch['packer']
        ob_clipart_tags = packer.ob_from_b(b_clipart_tags)
        ob_clipart_tags = self.pre_attn_tag_dropout(ob_clipart_tags)
        ob_scene_mask = packer.ob_from_b(example_batch['b_scene_mask'])

        brw_teller_tokens_in = example_batch['brw_teller_tokens_in']
        if self.training:
            word_dropout_probs = 1. / (1. + example_batch['brw_teller_counts_in'])
            brw_word_dropout_mask = torch.rand_like(word_dropout_probs) < word_dropout_probs
            brw_teller_tokens_in = torch.where(brw_word_dropout_mask, torch.full_like(brw_teller_tokens_in, dg.unk_index), brw_teller_tokens_in)

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

        if self.prediction_loss_scale > 0:
            brw_starts_round = (example_batch['brw_teller_tokens_in'] == dg.vocabulary_dict['<S>'])
            if self.predict_for_full_library:
                br_clipart_state_predictor_in = torch.cat([
                    packer.brw_from_orwb_unpack(orwb_lstm_out)[brw_starts_round],
                    packer.br_from_b_expand(example_batch['b_scene_mask']).to(torch.float),
                    ], -1)
            else:
                br_clipart_state_predictor_in = packer.brw_from_orwb_unpack(orwb_lstm_out)[brw_starts_round]
            bri_clipart_state_logits = self.clipart_state_predictor(br_clipart_state_predictor_in).view(-1, dg.NUM_CLIPART_STATES)
            bri_clipart_state_losses = F.cross_entropy(bri_clipart_state_logits, example_batch['br_drawer_clipart_state'].view(-1), reduce=False)
            if self.predict_for_full_library:
                br_clipart_state_losses = bri_clipart_state_losses.view(-1, dg.NUM_INDEX).sum(-1)
            else:
                br_clipart_state_losses = torch.where(
                    packer.br_from_b_expand(example_batch['b_scene_mask']),
                    bri_clipart_state_losses.view(-1, dg.NUM_INDEX),
                    torch.zeros_like(bri_clipart_state_losses.view(-1, dg.NUM_INDEX))).sum(-1)

        if return_loss:
            # Label smoothing
            eps = (self.label_smoothing / brw_word_logits.shape[-1])
            brw_word_losses = (1. - self.label_smoothing) * brw_word_losses + eps * (-F.log_softmax(brw_word_logits, dim=-1).sum(dim=-1))

            # TODO(nikita): Packer should implement some reduction operations
            per_example_word_losses = nn.utils.rnn.pad_packed_sequence(packer.orwb_from_brw_pack(brw_word_losses))[0].sum(0)
            word_loss = per_example_word_losses.mean()

            if self.prediction_loss_scale > 0:
                per_example_prediction_losses = nn.utils.rnn.pad_packed_sequence(packer.srb_from_br_pack(br_clipart_state_losses))[0].sum(0)
                prediction_loss = per_example_prediction_losses.mean()

                return self.prediction_loss_scale * prediction_loss + word_loss
            else:
                return word_loss

        if return_nll_count:
            # TODO(nikita): the model uses multiple tokens to signal the end of
            # the last utterance, followed by the end of the conversation. These
            # extra actions make perplexity not quite the same as models that
            # do stop tokens differently
            brw_non_unk_mask = example_batch['brw_teller_tokens_out'] != dg.unk_index
            brw_nll = torch.where(brw_non_unk_mask, brw_word_losses, torch.zeros_like(brw_word_losses))
            nll = float(brw_nll.sum())
            count = int(brw_non_unk_mask.long().sum())
            return nll, count

        assert False, "unreachable"

    @respond_to(codraw_data.ObserveTruth)
    @respond_to(codraw_data.ReplyGroup)
    def tell(self, episode):
        if not hasattr(episode, 'to_tell'):
            self.prepare(episode)

        if episode.to_tell:
            events = episode.to_tell.pop(0)
            episode.extend(events)

    def prepare(self, episode):
        true_scene = episode.get_last(codraw_data.ObserveTruth).scene

        example_batch = self.datagen.tensors_from_episode(episode)
        b_scene_mask, ks_prelstm, vs_prelstm, ks, vs = self.forward(example_batch, return_loss=False)

        to_tell = []

        lstm_state = None # carried across conversation rounds!

        for round in range(self.max_rounds):
            tokens = [self.datagen.vocabulary_dict['<S>']]
            events_this_round = []
            # Longest utterance in all of CoDraw is 39 words
            # Humans have a 140-char limit, but this is not easy to enforce with
            # word-level tokenization
            for wordnum in range(50):
                token_emb = self.word_embs(torch.tensor(tokens[-1], dtype=torch.long).to(cuda_if_available))[None,None,:]
                attended_values_prelstm = self.attn_prelstm(token_emb, ks=ks_prelstm, vs=vs_prelstm, k_mask=b_scene_mask)
                lstm_in = torch.cat([token_emb, attended_values_prelstm], -1)
                lstm_out, lstm_state = self.lstm(lstm_in, lstm_state)
                attended_values = self.attn(lstm_out, ks=ks, vs=vs, k_mask=b_scene_mask)
                pre_project = torch.cat([lstm_out, attended_values], -1)

                if tokens[-1] == self.datagen.vocabulary_dict['<S>'] and self.prediction_loss_scale > 0:
                    assert not events_this_round
                    if self.predict_for_full_library:
                        clipart_state_predictor_in = torch.cat([
                            lstm_out,
                            b_scene_mask.to(torch.float)[None,:,:],
                            ], -1)
                    else:
                        clipart_state_predictor_in = lstm_out
                    clipart_state_logits = self.clipart_state_predictor(clipart_state_predictor_in).view(self.datagen.NUM_INDEX, self.datagen.NUM_CLIPART_STATES)
                    clipart_state_selected = clipart_state_logits.argmax(dim=-1)
                    undrawn = AbstractScene([c for c in true_scene if clipart_state_selected[c.idx] == self.datagen.CLIPART_STATE_UNDRAWN])
                    intention = codraw_data.TellerIntention(drawn=None, undrawn=undrawn, draw_next=None)
                    events_this_round.append(intention)

                word_logits = self.word_project(pre_project[0,0,:])
                word_logits[self.datagen.vocabulary_dict['<S>']] = -float('inf')
                if round == 0 and wordnum == 0:
                    word_logits[self.datagen.vocabulary_dict['</TELL>']] = -float('inf')

                if self.inference_method == 'greedy':
                    next_token = int(word_logits.argmax())
                elif self.inference_method == 'sample':
                    next_token = int(torch.multinomial(F.softmax(word_logits / self.sampling_temperature, dim=-1)[None, :], 1).item())
                else:
                    raise ValueError(f"Invalid inference_method: {self.inference_method}")

                assert next_token != self.datagen.vocabulary_dict['<S>']
                tokens.append(next_token)
                if next_token == self.datagen.vocabulary_dict['</S>']:
                    break
                elif next_token == self.datagen.vocabulary_dict['</TELL>']:
                    break

            if tokens[-1] == self.datagen.vocabulary_dict['</TELL>']:
                break

            msg = " ".join([self.datagen.vocabulary[i] for i in tokens[1:-1]])
            events_this_round.append(codraw_data.TellGroup(msg))
            to_tell.append(events_this_round)

        episode.to_tell = to_tell

    def get_action_fns(self):
        return [self.tell]

    def calc_split_loss(self, split='dev'):
        """
        Calculates teller loss on a full split
        """
        datagen_spec = {**self.datagen.spec}
        datagen_spec['split'] = split
        datagen_dev = self.datagen_cls(spec=datagen_spec)

        assert datagen_dev.vocabulary == self.datagen.vocabulary

        losses = []
        count = 0
        with torch.no_grad():
            self.eval()
            for ex in datagen_dev.get_examples_unshuffled_batch(batch_size=128):
                batch_size = ex['b_scene_mask'].shape[0]
                loss = self.forward(ex)
                loss = float(loss) * batch_size
                losses.append(loss)
                count += batch_size

        return np.array(losses).sum() / count

# %%

def load_baseline3():
    baseline3_specs = torch_load(Path('models/scene2seq_july11.pt'))

    models = {}
    for k, spec in baseline3_specs.items():
        print(k)
        models[k] = globals()[spec['class']](spec=spec)
        models[k].eval()

    return models
