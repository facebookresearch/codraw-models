# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
An event-based view of the CoDraw dataset
"""

#%%
import numpy as np

from pathlib import Path
import json
from enum import Enum
from collections import namedtuple
import inspect

import abs_util_orig
import abs_render

#%%

if INTERACTIVE:
    DATASET_PATH = Path('../CoDraw/dataset/CoDraw_1_0.json')
else:
    DATASET_PATH = Path(__file__).parent / '../CoDraw/dataset/CoDraw_1_0.json'

assert DATASET_PATH.exists()

#%% clipart wrappers, with better docs than abs_util_orig.py

ClipartBase = namedtuple('Clipart',
        ['idx', 'subtype', 'depth', 'flip', 'x', 'y'])
	# idx: integer [0-57]
	# subtype: integer [0-34]
	# depth: integer [0-2]
	# flip: integer [0-1]
	# x: float [1-500]
	# y: float [1-400]

class Clipart(ClipartBase):
    __slots__ = ()
    NUM_IDX = 58
    NUM_SUBTYPE = 35
    NUM_DEPTH = 3
    NUM_FLIP = 2
    CANVAS_WIDTH = 500.0
    CANVAS_HEIGHT = 400.0

    NUM_EXPRESSION = 5
    NUM_POSE = 7
    assert NUM_SUBTYPE == (NUM_EXPRESSION * NUM_POSE)

    HUMAN_IDXS = (18, 19)

    def __new__(cls, idx, subtype, depth, flip, x=None, y=None, normed_x=None, normed_y=None):
        if normed_x is not None:
            if x is not None:
                raise ValueError("The arguments x and normed_x are mutually exclusive")
            x = normed_x * cls.CANVAS_WIDTH
        elif x is None:
            raise ValueError("Either x or normed_x is required")
        if normed_y is not None:
            if y is not None:
                raise ValueError("The arguments y and normed_y are mutually exclusive")
            y = normed_y * cls.CANVAS_HEIGHT
        elif y is None:
            raise ValueError("Either y or normed_y is required")

        return ClipartBase.__new__(cls, idx, subtype, depth, flip, x, y)

    @property
    def normed_x(self):
        return self.x / self.CANVAS_WIDTH

    @property
    def normed_y(self):
        return self.y / self.CANVAS_HEIGHT

    @property
    def expression(self):
        """
        Facial expression
        """
        return self.subtype % self.NUM_EXPRESSION

    @property
    def pose(self):
        """
        Body pose
        """
        return self.subtype // self.NUM_EXPRESSION

    @property
    def human_idx(self):
        if self.idx not in self.HUMAN_IDXS:
            raise ValueError("Cannot get human_idx of non-human clipart")
        return self.idx - self.HUMAN_IDXS[0]

    @property
    def render_order_key(self):
        """
        Key that can be used to sort cliparts by the order in which they are
        rendered.
        """
        # Sun (idx=3) is always in the back; this is also in Abs.js
        # All sky objects (idx < 8) are behind any non-sky objects
        # Past that, objects are sorted by depth and then by index
        return (self.idx != 3, self.idx >= 8, -self.depth, self.idx)

    def _repr_svg_(self):
        return abs_render.svg_from_cliparts([self])

class AbstractScene(list):
    """
    Abstract scene representation that only encodes objects which are present,
    and never a library of available objects that are not in the scene
    """
    def __init__(self, string_or_iterable):
        if isinstance(string_or_iterable, str):
            abs = abs_util_orig.AbsUtil(string_or_iterable)
            if abs.obj is None:
                super().__init__()
            else:
                super().__init__(Clipart(*c) for c in abs.obj)
        else:
            super().__init__(string_or_iterable)

    def __repr__(self):
        return "<AbstractScene " + super().__repr__() + ">"

    def __str__(self):
        return super().__repr__()

    def _repr_svg_(self):
        return abs_render.svg_from_cliparts(self)

    def stringify(self):
        scene_str = ""
        scene_str += f"{len(self)},"
        for i, clipart in enumerate(self):
            img_name = abs_render.get_image_name(clipart)
            prefix, num = img_name[:-5].split('_')
            prefix = ['s', 'p', 'hb0', 'hb1', 'a', 'c', 'e', 't'].index(prefix)
            num = int(num)

            scene_str += f"{img_name},"
            scene_str += f"{i},"
            scene_str += f"{num},"
            scene_str += f"{prefix},"
            scene_str += f"{clipart.x},"
            scene_str += f"{clipart.y},"
            scene_str += f"{clipart.depth},"
            scene_str += f"{clipart.flip},"
        return scene_str


#%% Data loading helper for a particular split

def data_for_splits(split_or_splits):
    if isinstance(split_or_splits, str):
        splits = [split_or_splits]
    else:
        splits = split_or_splits

    data_all = json.loads(DATASET_PATH.read_text())['data']
    keys_train = sorted([k for k in data_all.keys() if k.startswith('train')])
    keys_dev = sorted([k for k in data_all.keys() if k.startswith('val')])
    keys_test = sorted([k for k in data_all.keys() if k.startswith('test')])
    keys_all = sorted(data_all.keys())


    half_train_len = len(keys_train) // 2
    keys_from_split = {
        'train_a': keys_train[:half_train_len],
        'a': keys_train[:half_train_len],
        'train_b': keys_train[half_train_len:],
        'b': keys_train[half_train_len:],
        'train_full': keys_train,
        'dev': keys_dev,
        'test': keys_test,
        'all': keys_all,
    }

    res = []
    for split in splits:
        data_split = {k: data_all[k] for k in keys_from_split[split]}
        res.append(data_split)

    return res

def cached_split_wrapper(fn):
    """
    Modifies the function to accept a split or list of splits instead of a
    a raw data dictionary for a single split, and caches results so they don't
    have to be recalculated.
    """
    fn.split_to_results = {}
    def deco(split_or_splits):
        if isinstance(split_or_splits, str):
            splits = [split_or_splits]
        else:
            splits = split_or_splits

        uncached_splits = [split for split in splits if split not in fn.split_to_results]
        uncached_splits_data = data_for_splits(uncached_splits)
        for split, data in zip(uncached_splits, uncached_splits_data):
            result = fn(data)
            if inspect.isgenerator(result):
                result = list(result)
            fn.split_to_results[split] = result

        if isinstance(split_or_splits, str):
            return fn.split_to_results[split_or_splits]
        else:
            return [fn.split_to_results[split] for split in split_or_splits]
    return deco

#%% An event-based view of the CoDraw dataset

# TODO(nikita): Agent class and actor/observer are currently doing nothing.
# Is there a need for them?

class Agent(Enum):
    TELLER = 0
    DRAWER = 1

class Event:
    def __init__(self, actor=None, observer=None):
        self.actor = actor
        self.observer = observer

class ObserveTruth(Event):
    def __init__(self, scene):
        super().__init__(observer=Agent.TELLER)
        self.scene = scene

    def __repr__(self):
        return f"{type(self).__name__}()"

class SelectClipart(Event):
    def __init__(self, clipart):
        super().__init__(actor=Agent.TELLER, observer=None)
        self.clipart = clipart

    def __repr__(self):
        return f"{type(self).__name__}(clipart={self.clipart})"

class TellerIntention(Event):
    def __init__(self, drawn=None, undrawn=None, draw_next=None):
        super().__init__(actor=Agent.TELLER, observer=None)
        self.drawn = drawn
        self.undrawn = undrawn
        self.draw_next = draw_next

    def __repr__(self):
        return f"{type(self).__name__}(drawn={self.drawn}, undrawn={self.undrawn}, draw_next={self.draw_next})"

class TellGroup(Event):
    # group because each word is an action
    def __init__(self, msg):
        super().__init__(actor=Agent.TELLER, observer=Agent.DRAWER)
        self.msg = msg

    def __repr__(self):
        return f"{type(self).__name__}(msg={repr(self.msg)})"

class Peek(Event):
    def __init__(self):
        super().__init__(actor=Agent.TELLER, observer=None)

    def __repr__(self):
        return f"{type(self).__name__}()"

class TellerObserveCanvas(Event):
    def __init__(self, scene):
        super().__init__(observer=Agent.TELLER)
        if not isinstance(scene, AbstractScene):
            scene = AbstractScene(scene)
        self.scene = scene

    def __repr__(self):
        return f"{type(self).__name__}({self.scene})"

class ObserveCanvas(Event):
    def __init__(self, scene):
        super().__init__(observer=Agent.DRAWER)
        if not isinstance(scene, AbstractScene):
            scene = AbstractScene(scene)
        self.scene = scene

    def __repr__(self):
        return f"{type(self).__name__}({self.scene})"

class DrawClipart(Event):
    # Draws or moves a clipart
    # Since multiple copies of the same clipart are not allowed, duplicate draw
    # events with the same id will result in the removal of the older instance
    # of the clipart to make way for the new one.
    def __init__(self, clipart):
        super().__init__(actor=Agent.DRAWER, observer=None)
        self.clipart = clipart

    def __repr__(self):
        return f"{type(self).__name__}(clipart={self.clipart})"

class DrawGroup(Event):
    # Draws or moves multiple (or no) cliparts at the same time
    # Since multiple copies of the same clipart are not allowed, duplicate draw
    # events with the same id will result in the removal of the older instance
    # of the clipart to make way for the new one.
    def __init__(self, cliparts):
        super().__init__(actor=Agent.DRAWER, observer=None)
        self.cliparts = cliparts

    def __repr__(self):
        return f"{type(self).__name__}(cliparts={self.cliparts})"

class SetDrawing(Event):
    # Updates the drawer canvas to exactly match the scene argumentt
    # This was added for transcripts of humans performing the task because
    # neither DrawClipart nor DrawGroup have support for removing clipart.
    def __init__(self, scene):
        super().__init__(actor=Agent.DRAWER, observer=None)
        self.scene = scene

    def __repr__(self):
        return f"{type(self).__name__}({self.scene})"

class ReplyGroup(Event):
    # group because each word is an action
    def __init__(self, msg):
        super().__init__(actor=Agent.DRAWER, observer=Agent.TELLER)
        self.msg = msg

    def __repr__(self):
        return f"{type(self).__name__}(msg={repr(self.msg)})"

#%%

def events_from_datum_place_one(datum):
    # TODO(nikita): this filtering keeps just over 25% of conversational rounds
    # What do I need to do to match the 37.6% number in the arxiv paper?
    # perhaps I should include the cases where a clipart is updated? But that
    # only seems to bring me up to around 31%
    buffer = []
    buffer.append(ObserveTruth(AbstractScene(datum['abs_t'])))

    for entry in datum['dialog']:
        abs_b = AbstractScene(entry['abs_b'])
        abs_d = AbstractScene(entry['abs_d'])

        strictly_additive = len(set(abs_b) - set(abs_d)) == 0
        added_cliparts = set(abs_d) - set(abs_b)
        if strictly_additive and len(added_cliparts) == 1 and entry['msg_t']:
            added_clipart = list(added_cliparts)[0]
            buffer.append(SelectClipart(added_clipart))
            buffer.append(TellGroup(entry['msg_t']))
            buffer.append(DrawClipart(added_clipart))
            buffer.append(ReplyGroup(entry['msg_d']))

    if isinstance(buffer[-1], ObserveTruth):
        return []
    return buffer

@cached_split_wrapper
def get_place_one(data):
    for datum in data.values():
        yield from events_from_datum_place_one(datum)

#%%

def events_from_datum_place_many(datum):
    buffer = []
    buffer.append(ObserveTruth(AbstractScene(datum['abs_t'])))

    for entry in datum['dialog']:
        abs_b = AbstractScene(entry['abs_b'])
        abs_d = AbstractScene(entry['abs_d'])

        added_cliparts = set(abs_d) - set(abs_b)
        added_cliparts = sorted(added_cliparts, key=lambda c: c.render_order_key)

        buffer.append(TellGroup(entry['msg_t']))
        buffer.append(DrawGroup(added_cliparts))
        buffer.append(ReplyGroup(entry['msg_d']))

    if isinstance(buffer[-1], ObserveTruth):
        return []
    return buffer

@cached_split_wrapper
def get_place_many(data):
    for datum in data.values():
        yield from events_from_datum_place_many(datum)

#%%

def events_from_datum_contextual_place_many(datum):
    buffer = []
    buffer.append(ObserveTruth(AbstractScene(datum['abs_t'])))

    for entry in datum['dialog']:
        abs_b = AbstractScene(entry['abs_b'])
        abs_d = AbstractScene(entry['abs_d'])

        added_cliparts = set(abs_d) - set(abs_b)
        added_cliparts = sorted(added_cliparts, key=lambda c: c.render_order_key)

        buffer.append(TellGroup(entry['msg_t']))
        buffer.append(ObserveCanvas(abs_b))
        buffer.append(DrawGroup(added_cliparts))
        buffer.append(ReplyGroup(entry['msg_d']))

    if isinstance(buffer[-1], ObserveTruth):
        return []
    return buffer

@cached_split_wrapper
def get_contextual_place_many(data):
    for datum in data.values():
        yield from events_from_datum_contextual_place_many(datum)

# %%

def events_from_datum_set_clipart(datum):
    buffer = []
    buffer.append(ObserveTruth(AbstractScene(datum['abs_t'])))

    for entry in datum['dialog']:
        abs_b = AbstractScene(entry['abs_b'])
        abs_d = AbstractScene(entry['abs_d'])

        buffer.append(TellGroup(entry['msg_t']))
        buffer.append(ObserveCanvas(abs_b))
        buffer.append(SetDrawing(abs_d))
        buffer.append(ReplyGroup(entry['msg_d']))

    if isinstance(buffer[-1], ObserveTruth):
        return []
    return buffer

@cached_split_wrapper
def get_set_clipart(data):
    for datum in data.values():
        yield from events_from_datum_set_clipart(datum)

# %%

def events_from_datum_set_clipart_pre_peek(datum):
    buffer = []
    buffer.append(ObserveTruth(AbstractScene(datum['abs_t'])))

    for entry in datum['dialog']:
        if entry.get('peeked', False):
            # Note that Peek happens before TellGroup
            break

        abs_b = AbstractScene(entry['abs_b'])
        abs_d = AbstractScene(entry['abs_d'])

        buffer.append(TellGroup(entry['msg_t']))
        buffer.append(ObserveCanvas(abs_b))
        buffer.append(SetDrawing(abs_d))
        buffer.append(ReplyGroup(entry['msg_d']))

    if isinstance(buffer[-1], ObserveTruth):
        return []
    return buffer

@cached_split_wrapper
def get_set_clipart_pre_peek(data):
    for datum in data.values():
        yield from events_from_datum_set_clipart_pre_peek(datum)

# %%

@cached_split_wrapper
def get_scenes(data):
    for datum in data.values():
        yield AbstractScene(datum['abs_t'])

# %%

@cached_split_wrapper
def get_scenes_and_scripts(data):
    for datum in data.values():
        scene = AbstractScene(datum['abs_t'])
        script = []
        for entry in datum['dialog']:
            if entry.get('peeked', False):
                script.append(Peek())
                script.append(TellerObserveCanvas(AbstractScene(entry['abs_b'])))
            if entry['msg_t']:
                script.append(TellGroup(entry['msg_t']))
        yield (scene, script)

# %%

@cached_split_wrapper
def get_scenes_and_scripts_with_peek(data):
    for datum in data.values():
        scene = AbstractScene(datum['abs_t'])
        script = []
        have_peeked = False
        for entry in datum['dialog']:
            if entry.get('peeked', False):
                script.append(Peek())
                script.append(TellerObserveCanvas(AbstractScene(entry['abs_b'])))
                have_peeked = True
            if entry['msg_t']:
                script.append(TellGroup(entry['msg_t']))

        # Exclude events with no Peek action, or no messages sent afterwards
        if have_peeked and not isinstance(script[-1], TellerObserveCanvas):
            yield (scene, script)

# %%

@cached_split_wrapper
def get_truth_and_human_scenes(data):
    for datum in data.values():
        scene = AbstractScene(datum['abs_t'])
        scene_after = None
        for entry in datum['dialog']:
            scene_after = entry['abs_d']
        assert scene_after is not None
        scene_after = AbstractScene(scene_after)
        yield (scene, scene_after)

@cached_split_wrapper
def get_truth_and_human_scenes_pre_peek(data):
    for datum in data.values():
        scene = AbstractScene(datum['abs_t'])
        scene_after = None
        for entry in datum['dialog']:
            if entry.get('peeked', False):
                break
            scene_after = entry['abs_d']
        assert scene_after is not None
        scene_after = AbstractScene(scene_after)
        yield (scene, scene_after)

@cached_split_wrapper
def get_truth_and_human_scenes_with_js_scores(data):
    for datum in data.values():
        scene = AbstractScene(datum['abs_t'])
        scene_after = None
        score_after = None
        for entry in datum['dialog']:
            if entry.get('score', None) is not None:
                score_after = entry['score']
                scene_after = entry['abs_d']
        assert scene_after is not None
        assert score_after is not None
        scene_after = AbstractScene(scene_after)
        yield (scene, scene_after, score_after)
