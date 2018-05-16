# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Scene-level nearest-neighbor teller
"""

from interactivity import INTERACTIVE, try_magic, try_cd
try_cd('~/dev/drawmodel/nkcodraw')

#%%

import numpy as np
from pathlib import Path

import codraw_data
from codraw_data import AbstractScene, Clipart
import abs_render
from abs_metric import scene_similarity, clipart_similarity
from episode import Episode, Transcriber, respond_to

import model
from model import make_fns, eval_fns
from model import Model

from baseline2_models import load_baseline2

# %%

scenes_and_scripts_dev = codraw_data.get_scenes_and_scripts('dev')

transcribe = Transcriber(
    'exp28_scenenn.py' if INTERACTIVE else __file__,
    scenes_and_scripts=scenes_and_scripts_dev[::110],
    scenes_description="scenes_and_scripts_dev[::110]")

# %%

models_baseline2 = load_baseline2()

# %%

drawer_lstmaddonly_a = models_baseline2['drawer_lstmaddonly_a']
drawer_lstmaddonly_b = models_baseline2['drawer_lstmaddonly_b']

# %%

from datagen import Datagen
class SceneNearestNeighborData(Datagen):
    def init_full(self):
        self.build_dicts()

    def init_from_spec(self):
        self.build_dicts()

    def build_dicts(self):
        self.scene_to_msgs = {}

        # calculate events
        events = codraw_data.get_contextual_place_many(self.split)

        scene = None
        msgs = None

        it = iter(events)
        for event in it:
            if isinstance(event, codraw_data.ObserveTruth):
                if scene is not None and msgs is not None:
                    self.scene_to_msgs[tuple(scene)] = msgs
                scene = event.scene
                msgs = []
            elif isinstance(event, codraw_data.TellGroup):
                msgs.append(event.msg)

        if scene is not None and msgs is not None:
            self.scene_to_msgs[tuple(scene)] = msgs

# %%

class SceneNearestNeighborTeller(Model):
    datagen_cls = SceneNearestNeighborData

    def prepare(self, episode):
        scene = episode.get_last(codraw_data.ObserveTruth).scene
        best_similarity = -1
        best_msgs = []
        best_scene_tuple = None
        for cand_scene_tuple in self.datagen.scene_to_msgs:
            cand_sim = scene_similarity(cand_scene_tuple, scene)
            if cand_sim > best_similarity:
                best_similarity = cand_sim
                best_msgs = self.datagen.scene_to_msgs[cand_scene_tuple]
                best_scene_tuple = cand_scene_tuple

        # display(AbstractScene(scene))
        # display(AbstractScene(best_scene_tuple))
        # display(best_similarity)
        episode.to_tell = best_msgs[::] # make a copy!

    @respond_to(codraw_data.ObserveTruth)
    @respond_to(codraw_data.ReplyGroup)
    def tell(self, episode):
        if not hasattr(episode, 'to_tell'):
            self.prepare(episode)

        if episode.to_tell:
            msg = episode.to_tell.pop(0)
            episode.append(codraw_data.TellGroup(msg))


    def get_action_fns(self):
        return [self.tell]

# %%

data_scenenn_a = SceneNearestNeighborData('a')
data_scenenn_b = SceneNearestNeighborData('b')

# %%

teller_scenenn_a = SceneNearestNeighborTeller(data_scenenn_a)
teller_scenenn_b = SceneNearestNeighborTeller(data_scenenn_b)

# %%

# Episode.run(codraw_data.get_scenes('dev')[0], make_fns('aa', (teller_scenenn_a, teller_scenenn_b), (drawer_lstmaddonly_a, drawer_lstmaddonly_b))).display()

# %%
# %%
# %%

print()
print()
print("Final evaluation on full dev set")

# %%

for splits in ('aa', 'ab', 'ba', 'bb'):
    sims = eval_fns(make_fns(splits, (teller_scenenn_a, teller_scenenn_b), (drawer_lstmaddonly_a, drawer_lstmaddonly_b)), limit=None)
    print(splits, sims.mean())
# aa 1.3095491909624886
# ab 1.3115692170881366

# nohier aa 2.229799264350204
# nohier ab 2.255167911899865

# %%

for splits in ('ba', 'bb'):
    sims = eval_fns(make_fns(splits, (teller_scenenn_a, teller_scenenn_b), (drawer_lstmaddonly_a, drawer_lstmaddonly_b)), limit=None)
    print(splits, sims.mean())

# %%

transcribe("exp28_scenenn",
    aa=make_fns('aa', (teller_scenenn_a, teller_scenenn_b), (drawer_lstmaddonly_a, drawer_lstmaddonly_b)),
    ab=make_fns('ab', (teller_scenenn_a, teller_scenenn_b), (drawer_lstmaddonly_a, drawer_lstmaddonly_b)),
)

# %%

# hieraddonlyseq = dict(
#     drawer_hieraddonlyseq_a = drawer_hieraddonlyseq_a.spec,
#     drawer_hieraddonlyseq_b = drawer_hieraddonlyseq_b.spec,
# )

#%%

# torch.save(hieraddonlyseq, Path('models/hieraddonlyseq.pt'))

# %%
# %%
# %%
# %%
