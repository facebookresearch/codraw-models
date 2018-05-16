# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from interactivity import INTERACTIVE, try_magic, try_cd
try_cd('~/dev/drawmodel/nkcodraw')

#%%
import json
import numpy as np

import codraw_data
import model
from abs_metric import scene_similarity
from pathlib import Path

#%%

TRANSCRIPTS_PATH = Path('transcripts-eval-v1.json')
TRANSCRIPTS_SPLIT = 'test'

#%%

transcripts = json.loads(TRANSCRIPTS_PATH.read_text())

#%%

def get_transcript_results(transcripts):
    data = transcripts['data']
    for datum in data.values():
        model_name = datum['model_name']
        scene = codraw_data.AbstractScene(datum['abs_t'])
        scene_after = None
        for entry in datum['dialog']:
            scene_after = entry['abs_d']
        assert scene_after is not None
        scene_after = codraw_data.AbstractScene(scene_after)
        yield (model_name, scene, scene_after)

#%%

compontent_evaluator = model.ComponentEvaluator.get()

#%%

true_to_human = {}
for true_scene, human_scene in codraw_data.get_truth_and_human_scenes(TRANSCRIPTS_SPLIT):
    true_to_human[tuple(true_scene)] = human_scene

# %%

model_to_sims = {}
model_to_numer = {}
model_to_denom = {}
true_scenes_set = set()
for model_name, true_scene, reconstructed_scene in get_transcript_results(transcripts):
    if model_name not in model_to_sims:
        model_to_sims[model_name] = []
    if model_name not in model_to_numer:
        assert model_name not in model_to_denom
        model_to_numer[model_name] = []
        model_to_denom[model_name] = []
    model_to_sims[model_name].append(scene_similarity(reconstructed_scene, true_scene))
    numer, denom = compontent_evaluator.eval_scene(reconstructed_scene, true_scene)
    model_to_numer[model_name].append(numer)
    model_to_denom[model_name].append(denom)
    true_scenes_set.add(tuple(true_scene))

#%%

print("Model           \t Scene similarity")
for model_name, sims in model_to_sims.items():
    print(f"{model_name:17s}\t {np.array(sims).mean():.2f}")

sims = np.array([scene_similarity(true_to_human[scene], scene) for scene in true_scenes_set])
print(f"{'human':17s}\t {np.array(sims).mean():.2f}")

#%%
print()
print()
#%%

print("Model           \t  Dir   \t Expr(human)\t Pose(human)\t Depth  \t xy (sq.)\t x-only  \t y-only")
for model_name in model_to_numer:
    numer = model_to_numer[model_name]
    denom = model_to_denom[model_name]
    components = np.array(numer).sum(0) / np.array(denom).sum(0)
    components = 1.0 - (components - compontent_evaluator.human_scores) / (compontent_evaluator.baseline_scores - compontent_evaluator.human_scores)
    print(f"{model_name:17s}\t",  "\t".join(f"{num: .6f}" for num in components))

human_numer_denom = [compontent_evaluator.eval_scene(true_to_human[scene], scene) for scene in true_scenes_set]
components = np.array([x[0] for x in human_numer_denom]).sum(0) / np.array([x[1] for x in human_numer_denom]).sum(0)
components = 1.0 - (components - compontent_evaluator.human_scores) / (compontent_evaluator.baseline_scores - compontent_evaluator.human_scores)
print(f"{'human':17s}\t",  "\t".join(f"{num: .6f}" for num in components))


#%%
