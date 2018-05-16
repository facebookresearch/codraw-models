# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from pathlib import Path
import editdistance
from collections import Counter

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

from nkfb_util import logsumexp, cuda_if_available
from packer import Packer

import codraw_data
from codraw_data import AbstractScene, Clipart
import abs_render
from abs_metric import scene_similarity, clipart_similarity
from episode import Episode, respond_to, response_partial

#%%

class Datagen:
    # the spec contains summaries (like a vocab list), but the events are stored
    # as a pointer and not as the actual events dictionary. The events get
    # restored only if needed, (which shouldn't really be the case because saved
    # models won't need to be trained further.)
    def __init__(self, split=None, spec=None, **kwargs):
        self._examples_cache = None
        if spec is not None:
            self.split = spec['split']
            self.init_from_spec(**{k: v for (k,v) in spec.items() if k != 'split'})
        else:
            self.split = split
            self.init_full(**kwargs)

    def init_full(self):
        raise NotImplementedError("Subclasses should override this")

    def init_from_spec(self):
        raise NotImplementedError("Subclasses should override this")

    def calc_derived(self):
        pass

    def get_spec(self):
        return {}

    @property
    def spec(self):
        spec = self.get_spec()
        if 'split' not in spec:
            spec['split'] = self.split
        return spec

    def get_examples(self):
        raise NotImplementedError("Subclasses should override this")

    def collate(self, batch):
        raise NotImplementedError("Subclasses should override this")

    def get_examples_batch(self, batch_size=16):
        if self._examples_cache is None:
            self._examples_cache = list(self.get_examples())

        batch = []
        epoch_examples = self._examples_cache[:]
        np.random.shuffle(epoch_examples)
        for ex in epoch_examples:
            batch.append(ex)
            if len(batch) == batch_size:
                yield self.collate(batch)
                batch = []

    def get_examples_unshuffled_batch(self, batch_size=16):
        """
        Does not shuffle, and the last batch may contain less elements.
        Originally added for perplexity evaluation.
        """
        if self._examples_cache is None:
            self._examples_cache = list(self.get_examples())

        batch = []
        epoch_examples = self._examples_cache[:]
        for ex in epoch_examples:
            batch.append(ex)
            if len(batch) == batch_size:
                yield self.collate(batch)
                batch = []
        if batch:
            yield self.collate(batch)

#%%

class NearestNeighborData(Datagen):
    def init_full(self):
        self.build_dicts()

    def init_from_spec(self):
        self.build_dicts()

    def build_dicts(self):
        # calculate events
        events = codraw_data.get_place_one(self.split)

        self.msg_to_clipart = {}
        self.clipart_to_msg = {}

        it = iter(events)
        for event in it:
            if isinstance(event, codraw_data.SelectClipart):
                clipart = event.clipart
                event = next(it)
                assert isinstance(event, codraw_data.TellGroup)
                msg = event.msg
                event = next(it)
                assert isinstance(event, codraw_data.DrawClipart)
                event = next(it)
                assert isinstance(event, codraw_data.ReplyGroup)

                self.msg_to_clipart[msg] = clipart
                self.clipart_to_msg[clipart] = msg

#%%

class MessageSimilarityData(Datagen):
    def init_full(self):
        self.build_dicts()

        vocabulary = set()
        for msg in self.msg_to_clipart:
            vocabulary |= set(msg.split())
        self.vocabulary = sorted(vocabulary)
        self.vocabulary_dict = {item: num for num, item in enumerate(self.vocabulary)}

        self.calc_derived()

    def init_from_spec(self, vocabulary):
        self.build_dicts()
        self.vocabulary = vocabulary
        self.vocabulary_dict = {item: num for num, item in enumerate(self.vocabulary)}

    def get_spec(self):
        return dict(vocabulary=self.vocabulary)

    def build_dicts(self):
        events = codraw_data.get_place_one(self.split)

        self.msg_to_clipart = {}

        it = iter(events)
        for event in it:
            if isinstance(event, codraw_data.SelectClipart):
                clipart = event.clipart
                event = next(it)
                assert isinstance(event, codraw_data.TellGroup)
                msg = event.msg
                assert msg != ""
                event = next(it)
                assert isinstance(event, codraw_data.DrawClipart)
                event = next(it)
                assert isinstance(event, codraw_data.ReplyGroup)

                self.msg_to_clipart[msg] = clipart

    def calc_derived(self):
        self.all_msgs = list(self.msg_to_clipart.keys())
        assert "" not in self.all_msgs
        all_cliparts = [self.msg_to_clipart[msg] for msg in self.all_msgs]

        self.similarity_matrix = np.zeros((len(all_cliparts), len(all_cliparts)))
        for i in range(self.similarity_matrix.shape[0]):
            for j in range(i, self.similarity_matrix.shape[1]):
                self.similarity_matrix[i, j] = clipart_similarity(all_cliparts[i], all_cliparts[j])

        for i in range(self.similarity_matrix.shape[0]):
            for j in range(i):
                self.similarity_matrix[i, j] = self.similarity_matrix[j, i]

        # Never suggest the same sentence as both the input and a candidate
        for i in range(self.similarity_matrix.shape[0]):
            self.similarity_matrix[i, i] = -1

        matrix_good = self.similarity_matrix > 4.5
        matrix_bad = (self.similarity_matrix < 3.5) & (self.similarity_matrix >= 0)
        for i in range(matrix_good.shape[0]):
            if not matrix_good[i].any():
                matrix_good[i, self.similarity_matrix[i].argmax()] = True

        self.cands_good = np.zeros_like(self.similarity_matrix, dtype=int)
        self.cands_good_lens = np.zeros(self.cands_good.shape[0], dtype=int)
        self.cands_bad = np.zeros_like(self.similarity_matrix, dtype=int)
        self.cands_bad_lens = np.zeros(self.cands_bad.shape[0], dtype=int)

        where_good_i, where_good_j = np.where(matrix_good)
        for i in range(matrix_good.shape[0]):
            cands_good = where_good_j[where_good_i == i]
            self.cands_good_lens[i] = len(cands_good)
            self.cands_good[i,:len(cands_good)] = cands_good

        where_bad_i, where_bad_j = np.where(matrix_bad)
        unique_vals, unique_indices = np.unique(where_bad_i, return_index=True)
        assert (unique_vals == np.arange(self.cands_bad.shape[0])).all()
        for i in range(matrix_bad.shape[0]):
            start = unique_indices[i]
            if i == matrix_bad.shape[0] - 1:
                assert (where_bad_i[start:] == i).all()
                cands_bad = where_bad_j[start:]
            else:
                end = unique_indices[i+1]
                assert (where_bad_i[start:end] == i).all()
                cands_bad = where_bad_j[start:end]
            self.cands_bad_lens[i] = len(cands_bad)
            self.cands_bad[i,:len(cands_bad)] = cands_bad

    def get_candidates_for(self, i):
        good = np.random.choice(self.cands_good[i][:self.cands_good_lens[i]])
        bad = np.random.choice(self.cands_bad[i][:self.cands_bad_lens[i]], size=19)
        return (good, *bad)

    def get_examples(self):
        for i in np.random.permutation(self.cands_good.shape[0]):
            cands = self.get_candidates_for(i)
            idxs = (i, *cands)

            words = []
            offsets = []
            next_offset = 0
            for idx in idxs:
                offsets.append(next_offset)
                toks = [self.vocabulary_dict.get(tok, None) for tok in self.all_msgs[idx].split()]
                toks = [tok for tok in toks if tok is not None]
                words.extend(toks)
                next_offset += len(toks)

            yield {
                'words': torch.LongTensor(words),
                'offsets': torch.LongTensor(offsets)
            }

    def get_examples_batch(self, batch_size=16):
        batch = []
        for ex in self.get_examples():
            batch.append(ex)
            if len(batch) == batch_size:
                yield self.collate(batch)
                batch = []

    def collate(self, batch):
        offsets = [x['offsets'] for x in batch]
        extra = 0
        for i in range(len(offsets)):
            offsets[i] += extra
            extra += len(batch[i]['words'])

        return {
            'words': torch.cat([x['words'] for x in batch]).to(cuda_if_available),
            'offsets': torch.cat(offsets).to(cuda_if_available),
        }

#%%

def vocabulary_for_split(split, event_getter=codraw_data.get_place_one):
    vocabulary = set()

    it = iter(event_getter(split))
    for event in it:
        if isinstance(event, codraw_data.TellGroup):
            msg = event.msg
            vocabulary |= set(msg.split())

    return sorted(vocabulary)

def vocabulary_counter_for_split(split, event_getter=codraw_data.get_place_one):
    vocabulary = Counter()

    it = iter(event_getter(split))
    for event in it:
        if isinstance(event, codraw_data.TellGroup):
            msg = event.msg
            vocabulary.update(msg.split())

    return vocabulary

#%%

class BOWtoClipartData(Datagen):
    def init_full(self):
        self.vocabulary = vocabulary_for_split(self.split)
        self.vocabulary_dict = {item: num for num, item in enumerate(self.vocabulary)}

        self.calc_derived()

    def init_from_spec(self, vocabulary):
        self.vocabulary = vocabulary
        self.vocabulary_dict = {item: num for num, item in enumerate(self.vocabulary)}

    def get_spec(self):
        return dict(vocabulary=self.vocabulary)

    def get_examples(self):
        it = iter(codraw_data.get_place_one(self.split))
        for event in it:
            if isinstance(event, codraw_data.SelectClipart):
                clipart = event.clipart
                event = next(it)
                assert isinstance(event, codraw_data.TellGroup)
                msg = event.msg
                event = next(it)
                assert isinstance(event, codraw_data.DrawClipart)
                event = next(it)
                assert isinstance(event, codraw_data.ReplyGroup)
                clipart_index = torch.LongTensor(np.array(clipart.idx, dtype=int))
                clipart_categorical = torch.LongTensor([
                    clipart.subtype, clipart.depth, clipart.flip])
                clipart_numerical = torch.tensor([clipart.normed_x, clipart.normed_y], dtype=torch.float)

                msg_idxs = [self.vocabulary_dict.get(word, None) for word in msg.split()]
                msg_idxs = [idx for idx in msg_idxs if idx is not None]

                msg_idxs = torch.LongTensor(msg_idxs)
                example = {
                    'clipart_index': clipart_index,
                    'clipart_categorical': clipart_categorical,
                    'clipart_numerical': clipart_numerical,
                    'msg_idxs': msg_idxs,
                }
                yield example

    def collate(self, batch):
        offsets = np.cumsum([0] + [len(x['msg_idxs']) for x in batch])[:-1]

        return {
            'clipart_index': torch.stack([x['clipart_index'] for x in batch]).to(cuda_if_available),
            'clipart_categorical': torch.stack([x['clipart_categorical'] for x in batch]).to(cuda_if_available),
            'clipart_numerical': torch.stack([x['clipart_numerical'] for x in batch]).to(cuda_if_available),
            'msg_idxs': torch.cat([x['msg_idxs'] for x in batch]).to(cuda_if_available),
            'offsets': torch.tensor(offsets).to(cuda_if_available),
        }

#%%

class ClipartToSeqData(Datagen):
    NUM_INDEX = Clipart.NUM_IDX
    NUM_SUBTYPES = Clipart.NUM_SUBTYPE
    NUM_DEPTH = Clipart.NUM_DEPTH
    NUM_FLIP = Clipart.NUM_FLIP
    NUM_BINARY = NUM_INDEX + NUM_SUBTYPES + NUM_DEPTH + NUM_FLIP
    BINARY_OFFSETS = np.cumsum([0, NUM_INDEX, NUM_SUBTYPES, NUM_DEPTH])

    NUM_NUMERICAL = 2 # x, y

    def init_full(self):
        self.vocabulary = ['<S>', '</S>'] + vocabulary_for_split(self.split)
        self.vocabulary_dict = {item: num for num, item in enumerate(self.vocabulary)}

        self.calc_derived()

    def init_from_spec(self, vocabulary):
        self.vocabulary = vocabulary
        self.vocabulary_dict = {item: num for num, item in enumerate(self.vocabulary)}

    def get_spec(self):
        return dict(vocabulary=self.vocabulary)

    def get_examples(self):
        it = iter(codraw_data.get_place_one(self.split))
        for event in it:
            if isinstance(event, codraw_data.SelectClipart):
                clipart = event.clipart
                event = next(it)
                assert isinstance(event, codraw_data.TellGroup)
                msg = event.msg
                event = next(it)
                assert isinstance(event, codraw_data.DrawClipart)
                event = next(it)
                assert isinstance(event, codraw_data.ReplyGroup)

                x = clipart.normed_x
                y = clipart.normed_y

                clipart_numerical = torch.tensor([x, y], dtype=torch.float)
                clipart_binary = torch.zeros(self.NUM_BINARY)

                for val, offset in zip([clipart.idx, clipart.subtype, clipart.depth, clipart.flip], self.BINARY_OFFSETS):
                    clipart_binary[val + offset] = 1.

                msg_idxs = [self.vocabulary_dict['<S>']] + [self.vocabulary_dict.get(word, None) for word in msg.split()] + [self.vocabulary_dict['</S>']]
                msg_idxs = [idx for idx in msg_idxs if idx is not None]

                msg_idxs = torch.LongTensor(msg_idxs)
                example = {
                    'clipart_binary': clipart_binary,
                    'clipart_numerical': clipart_numerical,
                    'msg_idxs': msg_idxs,
                }
                yield example

    def collate(self, batch):
        batch = sorted(batch, key=lambda x: -len(x['msg_idxs']))

        msg_lens = torch.tensor([len(x['msg_idxs']) - 1 for x in batch], dtype=torch.long)
        max_len = int(msg_lens.max())
        msg_idxs_input = torch.stack([F.pad(torch.tensor(x['msg_idxs'][:-1]), (0, max_len + 1 - len(x['msg_idxs']))) for x in batch])
        msg_idxs_output = torch.stack([F.pad(torch.tensor(x['msg_idxs'][1:]), (0, max_len + 1 - len(x['msg_idxs']))) for x in batch])

        return {
            'clipart_binary': torch.stack([x['clipart_binary'] for x in batch]).to(cuda_if_available),
            'clipart_numerical': torch.stack([x['clipart_numerical'] for x in batch]).to(cuda_if_available),
            'msg_in': nn.utils.rnn.pack_padded_sequence(msg_idxs_input.to(cuda_if_available), msg_lens.to(cuda_if_available), batch_first=True),
            'msg_out': nn.utils.rnn.pack_padded_sequence(msg_idxs_output.to(cuda_if_available), msg_lens.to(cuda_if_available), batch_first=True),
        }

#%%

class BOWplusCanvasToMultiData(Datagen):
    NUM_INDEX = Clipart.NUM_IDX
    NUM_SUBTYPES = Clipart.NUM_SUBTYPE
    NUM_DEPTH = Clipart.NUM_DEPTH
    NUM_FLIP = Clipart.NUM_FLIP
    NUM_CATEGORICAL = NUM_SUBTYPES + NUM_DEPTH + NUM_FLIP
    NUM_NUMERICAL = 2 # x, y

    NUM_ALL = NUM_CATEGORICAL + NUM_NUMERICAL

    NUM_BINARY = (NUM_INDEX * (1 + NUM_DEPTH + NUM_FLIP)) + 2 * NUM_SUBTYPES

    def init_full(self):
        self.vocabulary = vocabulary_for_split(self.split, codraw_data.get_contextual_place_many)
        self.vocabulary_dict = {item: num for num, item in enumerate(self.vocabulary)}

        self.calc_derived()

    def init_from_spec(self, vocabulary):
        self.vocabulary = vocabulary
        self.vocabulary_dict = {item: num for num, item in enumerate(self.vocabulary)}

    def get_spec(self):
        return dict(vocabulary=self.vocabulary)

    def get_examples(self):
        it = iter(codraw_data.get_contextual_place_many(self.split))
        for event in it:
            if isinstance(event, codraw_data.TellGroup):
                assert isinstance(event, codraw_data.TellGroup)
                msg = event.msg
                event = next(it)
                assert isinstance(event, codraw_data.ObserveCanvas)
                canvas_context = event.scene
                event = next(it)
                assert isinstance(event, codraw_data.DrawGroup)
                cliparts = event.cliparts
                event = next(it)
                assert isinstance(event, codraw_data.ReplyGroup)

                if not msg:
                    continue

                clipart_chosen_mask = np.zeros(self.NUM_INDEX , dtype=bool)
                clipart_categorical = np.zeros((self.NUM_INDEX, 3))
                clipart_numerical = np.zeros((self.NUM_INDEX, self.NUM_NUMERICAL))
                for clipart in cliparts:
                    clipart_chosen_mask[clipart.idx] = True
                    clipart_categorical[clipart.idx, :] = [clipart.subtype, clipart.depth, clipart.flip]
                    clipart_numerical[clipart.idx, :] = [clipart.normed_x, clipart.normed_y]

                clipart_chosen_mask = torch.tensor(clipart_chosen_mask.astype(np.uint8), dtype=torch.uint8)
                clipart_categorical = torch.tensor(clipart_categorical, dtype=torch.long)
                clipart_numerical = torch.tensor(clipart_numerical, dtype=torch.float)

                canvas_binary = np.zeros((self.NUM_INDEX, 1 + self.NUM_DEPTH + self.NUM_FLIP), dtype=bool)
                canvas_pose = np.zeros((2, self.NUM_SUBTYPES), dtype=bool)
                canvas_numerical = np.zeros((self.NUM_INDEX, self.NUM_NUMERICAL))
                for clipart in canvas_context:
                    if clipart.idx in Clipart.HUMAN_IDXS:
                        canvas_pose[clipart.human_idx, clipart.subtype] = True

                    canvas_binary[clipart.idx, 0] = True
                    canvas_binary[clipart.idx, 1 + clipart.depth] = True
                    canvas_binary[clipart.idx, 1 + self.NUM_DEPTH + clipart.flip] = True
                    canvas_numerical[clipart.idx, 0] = clipart.normed_x
                    canvas_numerical[clipart.idx, 1] = clipart.normed_y

                canvas_binary = np.concatenate([canvas_binary.reshape((-1,)), canvas_pose.reshape((-1,))])
                canvas_numerical = canvas_numerical.reshape((-1,))

                canvas_binary = torch.tensor(canvas_binary.astype(np.uint8), dtype=torch.uint8)
                canvas_numerical = torch.tensor(canvas_numerical, dtype=torch.float)

                msg_idxs = [self.vocabulary_dict.get(word, None) for word in msg.split()]
                msg_idxs = [idx for idx in msg_idxs if idx is not None]

                msg_idxs = torch.LongTensor(msg_idxs)
                example = {
                    'clipart_chosen_mask': clipart_chosen_mask,
                    'clipart_categorical': clipart_categorical,
                    'clipart_numerical': clipart_numerical,
                    'canvas_binary': canvas_binary,
                    'canvas_numerical': canvas_numerical,
                    'msg_idxs': msg_idxs,
                }
                yield example

    def collate(self, batch):
        offsets = np.cumsum([0] + [len(x['msg_idxs']) for x in batch])[:-1]

        return {
            'clipart_chosen_mask': torch.stack([x['clipart_chosen_mask'] for x in batch]).to(cuda_if_available),
            'clipart_categorical': torch.stack([x['clipart_categorical'] for x in batch]).to(cuda_if_available),
            'clipart_numerical': torch.stack([x['clipart_numerical'] for x in batch]).to(cuda_if_available),
            'canvas_binary': torch.stack([x['canvas_binary'] for x in batch]).to(cuda_if_available),
            'canvas_numerical': torch.stack([x['canvas_numerical'] for x in batch]).to(cuda_if_available),
            'msg_idxs': torch.cat([x['msg_idxs'] for x in batch]).to(cuda_if_available),
            'offsets': torch.tensor(offsets).to(cuda_if_available),
        }

#%%


class BOWAddUpdateData(Datagen):
    NUM_INDEX = Clipart.NUM_IDX
    NUM_SUBTYPES = Clipart.NUM_SUBTYPE
    NUM_DEPTH = Clipart.NUM_DEPTH
    NUM_FLIP = Clipart.NUM_FLIP
    NUM_CATEGORICAL = NUM_SUBTYPES + NUM_DEPTH + NUM_FLIP
    NUM_NUMERICAL = 2 # x, y

    NUM_ALL = NUM_CATEGORICAL + NUM_NUMERICAL

    NUM_BINARY = (NUM_INDEX * (1 + NUM_DEPTH + NUM_FLIP)) + 2 * NUM_SUBTYPES

    NUM_X_TICKS = 3
    NUM_Y_TICKS = 2
    NUM_TAGS = 1 + Clipart.NUM_IDX + Clipart.NUM_SUBTYPE + Clipart.NUM_DEPTH + Clipart.NUM_FLIP + NUM_X_TICKS + NUM_Y_TICKS + 1
    NUM_TAGS_PER_INDEX = 6 # index, subtype, depth, flip, x, y

    def init_full(self):
        self.vocabulary = vocabulary_for_split(self.split, codraw_data.get_contextual_place_many)
        self.vocabulary_dict = {item: num for num, item in enumerate(self.vocabulary)}

        self.calc_derived()

    def init_from_spec(self, vocabulary):
        self.vocabulary = vocabulary
        self.vocabulary_dict = {item: num for num, item in enumerate(self.vocabulary)}

    def get_spec(self):
        return dict(vocabulary=self.vocabulary)

    def get_examples(self):
        it = iter(codraw_data.get_contextual_place_many(self.split))
        for event in it:
            if isinstance(event, codraw_data.TellGroup):
                assert isinstance(event, codraw_data.TellGroup)
                msg = event.msg
                event = next(it)
                assert isinstance(event, codraw_data.ObserveCanvas)
                canvas_context = event.scene
                event = next(it)
                assert isinstance(event, codraw_data.DrawGroup)
                cliparts = event.cliparts
                event = next(it)
                assert isinstance(event, codraw_data.ReplyGroup)

                if not msg:
                    continue

                context_idxs = set([c.idx for c in canvas_context])

                clipart_added_mask = np.zeros(self.NUM_INDEX , dtype=bool)
                clipart_updated_mask = np.zeros(self.NUM_INDEX , dtype=bool)
                clipart_categorical = np.zeros((self.NUM_INDEX, 3))
                clipart_numerical = np.zeros((self.NUM_INDEX, self.NUM_NUMERICAL))
                for clipart in cliparts:
                    if clipart.idx in context_idxs:
                        clipart_updated_mask[clipart.idx] = True
                    else:
                        clipart_added_mask[clipart.idx] = True
                    clipart_categorical[clipart.idx, :] = [clipart.subtype, clipart.depth, clipart.flip]
                    clipart_numerical[clipart.idx, :] = [clipart.normed_x, clipart.normed_y]

                clipart_added_mask = torch.tensor(clipart_added_mask.astype(np.uint8), dtype=torch.uint8)
                clipart_updated_mask = torch.tensor(clipart_updated_mask.astype(np.uint8), dtype=torch.uint8)
                clipart_categorical = torch.tensor(clipart_categorical, dtype=torch.long)
                clipart_numerical = torch.tensor(clipart_numerical, dtype=torch.float)

                canvas_binary = np.zeros((self.NUM_INDEX, 1 + self.NUM_DEPTH + self.NUM_FLIP), dtype=bool)
                canvas_pose = np.zeros((2, self.NUM_SUBTYPES), dtype=bool)
                canvas_numerical = np.zeros((self.NUM_INDEX, self.NUM_NUMERICAL))
                canvas_tags = np.zeros((self.NUM_INDEX + 1, self.NUM_TAGS_PER_INDEX), dtype=int)
                canvas_mask = np.zeros(self.NUM_INDEX + 1, dtype=bool)
                for clipart in canvas_context:
                    if clipart.idx in Clipart.HUMAN_IDXS:
                        canvas_pose[clipart.human_idx, clipart.subtype] = True

                    canvas_binary[clipart.idx, 0] = True
                    canvas_binary[clipart.idx, 1 + clipart.depth] = True
                    canvas_binary[clipart.idx, 1 + self.NUM_DEPTH + clipart.flip] = True
                    canvas_numerical[clipart.idx, 0] = clipart.normed_x
                    canvas_numerical[clipart.idx, 1] = clipart.normed_y

                    x_tick = int(np.floor(clipart.normed_x * self.NUM_X_TICKS))
                    if x_tick < 0:
                        x_tick = 0
                    elif x_tick >= self.NUM_X_TICKS:
                        x_tick = self.NUM_X_TICKS - 1

                    y_tick = int(np.floor(clipart.normed_y * self.NUM_Y_TICKS))
                    if y_tick < 0:
                        y_tick = 0
                    elif y_tick >= self.NUM_Y_TICKS:
                        y_tick = self.NUM_Y_TICKS - 1

                    # Tag features (for attention)
                    canvas_tags[clipart.idx, 0] = 1 + clipart.idx
                    canvas_tags[clipart.idx, 1] = 1 + Clipart.NUM_IDX + clipart.subtype
                    canvas_tags[clipart.idx, 2] = 1 + Clipart.NUM_IDX + Clipart.NUM_SUBTYPE + clipart.depth
                    canvas_tags[clipart.idx, 3] = 1 + Clipart.NUM_IDX + Clipart.NUM_SUBTYPE + Clipart.NUM_DEPTH + int(clipart.flip)
                    canvas_tags[clipart.idx, 4] = 1 + Clipart.NUM_IDX + Clipart.NUM_SUBTYPE + Clipart.NUM_DEPTH + Clipart.NUM_FLIP + x_tick
                    canvas_tags[clipart.idx, 5] = 1 + Clipart.NUM_IDX + Clipart.NUM_SUBTYPE + Clipart.NUM_DEPTH + Clipart.NUM_FLIP + self.NUM_X_TICKS + y_tick

                    canvas_mask[clipart.idx] = True

                if not canvas_context:
                    canvas_tags[-1, 0] = self.NUM_TAGS - 1
                    canvas_mask[-1] = True

                canvas_binary = np.concatenate([canvas_binary.reshape((-1,)), canvas_pose.reshape((-1,))])
                canvas_numerical = canvas_numerical.reshape((-1,))

                canvas_binary = torch.tensor(canvas_binary.astype(np.uint8), dtype=torch.uint8)
                canvas_numerical = torch.tensor(canvas_numerical, dtype=torch.float)

                canvas_tags = torch.tensor(canvas_tags, dtype=torch.long)
                canvas_mask = torch.tensor(canvas_mask.astype(np.uint8), dtype=torch.uint8)

                msg_idxs = [self.vocabulary_dict.get(word, None) for word in msg.split()]
                msg_idxs = [idx for idx in msg_idxs if idx is not None]

                msg_idxs = torch.LongTensor(msg_idxs)
                example = {
                    'clipart_added_mask': clipart_added_mask,
                    'clipart_updated_mask': clipart_updated_mask,
                    'clipart_categorical': clipart_categorical,
                    'clipart_numerical': clipart_numerical,
                    'canvas_binary': canvas_binary,
                    'canvas_numerical': canvas_numerical,
                    'canvas_tags': canvas_tags,
                    'canvas_mask': canvas_mask,
                    'msg_idxs': msg_idxs,
                }
                yield example

    def collate(self, batch):
        offsets = np.cumsum([0] + [len(x['msg_idxs']) for x in batch])[:-1]

        return {
            'clipart_added_mask': torch.stack([x['clipart_added_mask'] for x in batch]).to(cuda_if_available),
            'clipart_updated_mask': torch.stack([x['clipart_updated_mask'] for x in batch]).to(cuda_if_available),
            'clipart_categorical': torch.stack([x['clipart_categorical'] for x in batch]).to(cuda_if_available),
            'clipart_numerical': torch.stack([x['clipart_numerical'] for x in batch]).to(cuda_if_available),
            'canvas_binary': torch.stack([x['canvas_binary'] for x in batch]).to(cuda_if_available),
            'canvas_numerical': torch.stack([x['canvas_numerical'] for x in batch]).to(cuda_if_available),
            'canvas_tags': torch.stack([x['canvas_tags'] for x in batch]).to(cuda_if_available),
            'canvas_mask': torch.stack([x['canvas_mask'] for x in batch]).to(cuda_if_available),
            'msg_idxs': torch.cat([x['msg_idxs'] for x in batch]).to(cuda_if_available),
            'offsets': torch.tensor(offsets).to(cuda_if_available),
        }

#%%

class SceneToSeqData(Datagen):
    NUM_INDEX = Clipart.NUM_IDX
    NUM_SUBTYPES = Clipart.NUM_SUBTYPE
    NUM_DEPTH = Clipart.NUM_DEPTH
    NUM_FLIP = Clipart.NUM_FLIP
    NUM_X_TICKS = 3
    NUM_Y_TICKS = 2
    NUM_BINARY = (NUM_INDEX * (1 + NUM_DEPTH + NUM_FLIP + NUM_X_TICKS + NUM_Y_TICKS)) + 2 * NUM_SUBTYPES

    NUM_TAGS = 1 + Clipart.NUM_IDX + Clipart.NUM_SUBTYPE + Clipart.NUM_DEPTH + Clipart.NUM_FLIP + NUM_X_TICKS + NUM_Y_TICKS
    NUM_TAGS_PER_INDEX = 6 # index, subtype, depth, flip, x, y

    CLIPART_STATE_NOT_UNDRAWN = 0
    CLIPART_STATE_UNDRAWN = 1
    NUM_CLIPART_STATES = 2

    def init_full(self):
        self.vocabulary_counts = vocabulary_counter_for_split(self.split, codraw_data.get_set_clipart_pre_peek)
        self.vocabulary = ['</TELL>', '<S>', '</S>', '<UNK>'] + sorted(self.vocabulary_counts.keys())
        self.vocabulary_dict = {item: num for num, item in enumerate(self.vocabulary)}
        self.unk_index = self.vocabulary_dict['<UNK>']

        self.calc_derived()

    def init_from_spec(self, vocabulary, vocabulary_counts):
        self.vocabulary_counts = vocabulary_counts
        self.vocabulary = vocabulary
        self.vocabulary_dict = {item: num for num, item in enumerate(self.vocabulary)}
        self.unk_index = self.vocabulary_dict['<UNK>']

    def get_spec(self):
        return dict(vocabulary=self.vocabulary, vocabulary_counts=self.vocabulary_counts)

    def tensors_from_episode(self, episode, is_train=False):
        examples = list(self.get_examples(episode, is_train=is_train))
        if not examples:
            print(episode)
            assert len(examples) > 0, "Episode did not produce any examples"
        assert len(examples) == 1, "Episode should not produce multiple examples"
        return self.collate(examples, is_train=is_train)

    def tensors_from_episodes(self, episodes, is_train=True):
        events = []
        for episode in episodes:
            events.extend(episode)
        examples = list(self.get_examples(events, is_train=is_train))
        if not examples:
            print(episode)
            assert len(examples) > 0, "Episode did not produce any examples"
        return self.collate(examples, is_train=is_train)

    def get_examples(self, events=None, is_train=True):
        example = None
        scene_present_idxs = None
        prev_drawn_idxs = None
        num_unfilled_past = None

        if events is None:
            events = codraw_data.get_set_clipart_pre_peek(self.split)
        it = iter(events)
        for event in it:
            if isinstance(event, codraw_data.ObserveTruth):
                if example is not None:
                    # When doing RL, it's important that the batched data
                    # matches the decisions taken in step-by-step mode
                    # If an episode was cut off, don't include a </TELL> token
                    # All human conversations have less than 50 rounds
                    if len(example['teller_tokens_in']) < 50:
                        teller_tokens_stop = [self.vocabulary_dict[x] for x in ('<S>', '</TELL>')]
                        teller_tokens_stop = torch.tensor(teller_tokens_stop, dtype=torch.long)
                        example['teller_tokens_in'].append(teller_tokens_stop[:-1])
                        example['teller_tokens_out'].append(teller_tokens_stop[1:])
                        example['teller_counts_in'].append(torch.tensor([np.inf], dtype=torch.float))
                    else:
                        example['drawer_clipart_state'].pop()
                    yield example

                scene = event.scene
                scene_present_idxs = set([c.idx for c in scene])

                scene_tags = np.zeros((self.NUM_INDEX, self.NUM_TAGS_PER_INDEX), dtype=int)
                scene_mask = np.zeros(self.NUM_INDEX, dtype=bool)
                for clipart in scene:
                    x_tick = int(np.floor(clipart.normed_x * self.NUM_X_TICKS))
                    if x_tick < 0:
                        x_tick = 0
                    elif x_tick >= self.NUM_X_TICKS:
                        x_tick = self.NUM_X_TICKS - 1

                    y_tick = int(np.floor(clipart.normed_y * self.NUM_Y_TICKS))
                    if y_tick < 0:
                        y_tick = 0
                    elif y_tick >= self.NUM_Y_TICKS:
                        y_tick = self.NUM_Y_TICKS - 1

                    # Tag features (for attention)
                    scene_tags[clipart.idx, 0] = 1 + clipart.idx
                    scene_tags[clipart.idx, 1] = 1 + Clipart.NUM_IDX + clipart.subtype
                    scene_tags[clipart.idx, 2] = 1 + Clipart.NUM_IDX + Clipart.NUM_SUBTYPE + clipart.depth
                    scene_tags[clipart.idx, 3] = 1 + Clipart.NUM_IDX + Clipart.NUM_SUBTYPE + Clipart.NUM_DEPTH + int(clipart.flip)
                    scene_tags[clipart.idx, 4] = 1 + Clipart.NUM_IDX + Clipart.NUM_SUBTYPE + Clipart.NUM_DEPTH + Clipart.NUM_FLIP + x_tick
                    scene_tags[clipart.idx, 5] = 1 + Clipart.NUM_IDX + Clipart.NUM_SUBTYPE + Clipart.NUM_DEPTH + Clipart.NUM_FLIP + self.NUM_X_TICKS + y_tick

                    scene_mask[clipart.idx] = True

                scene_tags = torch.tensor(scene_tags, dtype=torch.long)
                scene_mask = torch.tensor(scene_mask.astype(np.uint8), dtype=torch.uint8)

                if is_train:
                    assert scene_present_idxs is not None
                    drawer_clipart_state = np.zeros(self.NUM_INDEX, dtype=int)
                    for idx in range(self.NUM_INDEX):
                        if idx not in scene_present_idxs:
                            # drawer_clipart_state[idx] = self.CLIPART_STATE_NOT_IN_SCENE
                            drawer_clipart_state[idx] = self.CLIPART_STATE_NOT_UNDRAWN
                        else:
                            drawer_clipart_state[idx] = self.CLIPART_STATE_UNDRAWN
                    drawer_clipart_state = torch.tensor(drawer_clipart_state, dtype=torch.long)
                    prev_drawn_idxs = set()
                    num_unfilled_past = 1

                    example = {
                        'scene_tags': scene_tags,
                        'scene_mask': scene_mask,
                        'teller_tokens_in': [],
                        'teller_counts_in': [],
                        'teller_tokens_out': [],
                        'drawer_clipart_state': [drawer_clipart_state],
                    }
                else:
                    yield {
                        'scene_tags': scene_tags,
                        'scene_mask': scene_mask,
                    }
                    # At test time, there shouldn't be anything after the
                    # ObserveTruth event
                    continue

            if isinstance(event, codraw_data.TellGroup):
                assert isinstance(event, codraw_data.TellGroup)
                msg = event.msg
                event = next(it)
                assert isinstance(event, codraw_data.ObserveCanvas)
                canvas_context = event.scene
                event = next(it)
                assert isinstance(event, codraw_data.SetDrawing)
                drawn_scene = event.scene
                event = next(it)
                assert isinstance(event, codraw_data.ReplyGroup)

                teller_tokens = [self.vocabulary_dict.get(word, self.unk_index) for word in msg.split()]
                teller_counts = [self.vocabulary_counts[word] for word in msg.split()]
                teller_tokens = [self.vocabulary_dict['<S>']] + teller_tokens + [self.vocabulary_dict['</S>']]
                teller_counts = [np.inf] + teller_counts + [np.inf]

                # Needed for RL. All human utterances have less than 50 words
                # due to a character limit imposed during data collection
                if len(teller_tokens) > 51:
                    teller_tokens = teller_tokens[:51]
                    teller_counts = teller_counts[:51]

                teller_tokens = torch.tensor(teller_tokens, dtype=torch.long)
                teller_counts = torch.tensor(teller_counts, dtype=torch.float)
                example['teller_tokens_in'].append(teller_tokens[:-1])
                example['teller_tokens_out'].append(teller_tokens[1:])
                example['teller_counts_in'].append(teller_counts[:-1])

                assert scene_present_idxs is not None
                drawn_idxs = set([c.idx for c in drawn_scene])

                drawer_clipart_state = np.zeros(self.NUM_INDEX, dtype=int)
                for idx in range(self.NUM_INDEX):
                    if idx not in scene_present_idxs or idx in drawn_idxs:
                        drawer_clipart_state[idx] = self.CLIPART_STATE_NOT_UNDRAWN
                    else:
                        drawer_clipart_state[idx] = self.CLIPART_STATE_UNDRAWN

                drawer_clipart_state = torch.tensor(drawer_clipart_state, dtype=torch.long)
                example['drawer_clipart_state'].append(drawer_clipart_state)

    def collate(self, batch, is_train=True):
        if is_train:
            packer = Packer([x['teller_tokens_in'] for x in batch])
            return {
                'packer': packer,
                'brw_teller_tokens_in': packer.brw_from_list([x['teller_tokens_in'] for x in batch]).to(cuda_if_available),
                'brw_teller_counts_in': packer.brw_from_list([x['teller_counts_in'] for x in batch]).to(cuda_if_available),
                'brw_teller_tokens_out': packer.brw_from_list([x['teller_tokens_out'] for x in batch]).to(cuda_if_available),
                'b_scene_tags': torch.stack([x['scene_tags'] for x in batch]).to(cuda_if_available),
                'b_scene_mask': torch.stack([x['scene_mask'] for x in batch]).to(cuda_if_available),
                'br_drawer_clipart_state': packer.br_from_list([x['drawer_clipart_state'] for x in batch]).to(cuda_if_available),
            }
        else:
            return {
                'b_scene_tags': torch.stack([x['scene_tags'] for x in batch]).to(cuda_if_available),
                'b_scene_mask': torch.stack([x['scene_mask'] for x in batch]).to(cuda_if_available),
            }
