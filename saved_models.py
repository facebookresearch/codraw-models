# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#%%

def load_models(*partitions):
    if not partitions:
        partitions = (1, 2, 3, 4)

    models = {}

    if 1 in partitions:
        from baseline1_models import load_baseline1
        models.update(load_baseline1())
    if 2 in partitions:
        from baseline2_models import load_baseline2
        models.update(load_baseline2())
    if 3 in partitions:
        from baseline3_models import load_baseline3
        models.update(load_baseline3())
    if 4 in partitions:
        from baseline4_models import load_baseline4
        models.update(load_baseline4())

    return models

#%%

def make_pairs(models, *names):
    if models is None:
        models = load_models()

    res = []
    for name in names:
        res.append((name, (models[name + '_a'], models[name + '_b'])))

    return res
