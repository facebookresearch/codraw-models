# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

def scene_similarity_orig(pred, target):
    """
    DEPRECATED: use scene_similarity instead!

    This is a re-implementation of the original CoDraw similarity metric, as per
    https://arxiv.org/abs/1712.05558v1
    """
    idx1 = set(x.idx for x in target)
    idx2 = set(x.idx for x in pred)
    iou = len(idx1 & idx2) / len(idx1 | idx2)

    common_idxs = list(idx1 & idx2)
    match1 = [[x for x in target if x.idx == idx][0] for idx in common_idxs]
    match2 = [[x for x in pred if x.idx == idx][0] for idx in common_idxs]

    num = np.zeros(7)
    denom = np.zeros(7)

    num[0] = 1

    for c1, c2 in zip(match1, match2):
        if c1.idx not in c1.HUMAN_IDXS:
            num[1] += int(c1.flip != c2.flip)
            denom[1] += 1
        else:
            num[2] += int(c1.subtype != c2.subtype or c1.flip != c2.flip)
            denom[2] += 1

        num[3] += int(c1.depth != c2.depth)
        num[4] += np.sqrt((c1.normed_x - c2.normed_x) ** 2 + (c1.normed_y - c2.normed_y) ** 2)
        denom[3] += 1
        denom[4] += 1

    for idx_i in range(len(match1)):
        for idx_j in range(len(match1)):
            c1i, c1j = match1[idx_i], match1[idx_j]
            c2i, c2j = match2[idx_i], match2[idx_j]

            # NOTE(nikita): the metric, as originally defined, pairs up objects
            # with themselves, and also yields misleadingly high results for
            # models that place multiple clipart at the exact same location
            # (e.g. a model that places all clipart in the center of the canvas
            # will receive zero relative-position penalty)
            num[5] += int((c1i.x - c1j.x) * (c2i.x - c2j.x) < 0)
            num[6] += int((c1i.y - c1j.y) * (c2i.y - c2j.y) < 0)
            denom[5] += 1
            denom[6] += 1

    denom = np.maximum(denom, 1)

    score_components = iou * (num / denom)
    score_weights = np.array([5,-1,-1,-1,-1,-0.5,-0.5])

    return score_components @ score_weights

def scene_similarity_v1(pred, target):
    """
    DEPRECATED: use scene_similarity instead!

    The similarity metric used for initial experiments prior to June 8, 2018.
    Both this metric and scene_similarity_orig have corner cases where adding a
    new, correct clipart to the scene can actually cause the similarity score
    to decrease.
    """
    idx1 = set(x.idx for x in target)
    idx2 = set(x.idx for x in pred)
    iou = len(idx1 & idx2) / len(idx1 | idx2)

    common_idxs = list(idx1 & idx2)
    match1 = [[x for x in target if x.idx == idx][0] for idx in common_idxs]
    match2 = [[x for x in pred if x.idx == idx][0] for idx in common_idxs]

    num = np.zeros(7)
    denom = np.zeros(7)

    num[0] = 1

    for c1, c2 in zip(match1, match2):
        if c1.idx not in c1.HUMAN_IDXS:
            num[1] += int(c1.flip != c2.flip)
            denom[1] += 1
        else:
            num[2] += int(c1.subtype != c2.subtype or c1.flip != c2.flip)
            denom[2] += 1

        num[3] += int(c1.depth != c2.depth)
        num[4] += np.sqrt((c1.normed_x - c2.normed_x) ** 2 + (c1.normed_y - c2.normed_y) ** 2)
        denom[3] += 1
        denom[4] += 1

    for idx_i in range(len(match1)):
        for idx_j in range(idx_i, len(match1)):
            if idx_i == idx_j:
                continue
            c1i, c1j = match1[idx_i], match1[idx_j]
            c2i, c2j = match2[idx_i], match2[idx_j]

            # TODO(nikita): this doesn't correctly handle the case if two
            # cliparts have *exactly* the same x/y coordinates in the target
            num[5] += int((c1i.x - c1j.x) * (c2i.x - c2j.x) <= 0)
            num[6] += int((c1i.y - c1j.y) * (c2i.y - c2j.y) <= 0)
            denom[5] += 1
            denom[6] += 1

    denom = np.maximum(denom, 1)

    score_components = iou * (num / denom)
    score_weights = np.array([5,-1,-1,-1,-1,-0.5,-0.5])

    return score_components @ score_weights


def scene_similarity_v2(pred, target):
    """
    DEPRECATED: use scene_similarity instead!

    This version of the scene similarity metric should be monotonic, in the
    sense that adding correct clipart should always increase the score, adding
    incorrect clipart should decrease it, and removing incorrect clipart should
    increase it.

    This version jointly scores subtype/flip/depth for humans, which was later
    replaced with a more fine-grained scoring
    """
    idx1 = set(x.idx for x in target)
    idx2 = set(x.idx for x in pred)
    iou = len(idx1 & idx2) / len(idx1 | idx2)

    intersection_size = len(idx1 & idx2)
    union_size = len(idx1 | idx2)

    common_idxs = list(idx1 & idx2)
    match1 = [[x for x in target if x.idx == idx][0] for idx in common_idxs]
    match2 = [[x for x in pred if x.idx == idx][0] for idx in common_idxs]

    num = np.zeros(7)
    denom = np.zeros(7)

    num[0] = intersection_size

    for c1, c2 in zip(match1, match2):
        if c1.idx not in c1.HUMAN_IDXS:
            num[1] += int(c1.flip != c2.flip)
        else:
            num[2] += int(c1.subtype != c2.subtype or c1.flip != c2.flip)
        num[3] += int(c1.depth != c2.depth)
        num[4] += np.sqrt((c1.normed_x - c2.normed_x) ** 2 + (c1.normed_y - c2.normed_y) ** 2)

    denom[:5] = union_size

    for idx_i in range(len(match1)):
        for idx_j in range(idx_i, len(match1)):
            if idx_i == idx_j:
                continue
            c1i, c1j = match1[idx_i], match1[idx_j]
            c2i, c2j = match2[idx_i], match2[idx_j]

            # TODO(nikita): this doesn't correctly handle the case if two
            # cliparts have *exactly* the same x/y coordinates in the target
            num[5] += int((c1i.x - c1j.x) * (c2i.x - c2j.x) <= 0)
            num[6] += int((c1i.y - c1j.y) * (c2i.y - c2j.y) <= 0)

    denom[5:] = union_size * (intersection_size - 1)

    denom = np.maximum(denom, 1)

    score_components = num / denom
    score_weights = np.array([5,-1,-1,-1,-1,-1,-1])

    return score_components @ score_weights


def scene_similarity(pred, target):
    """
    This version of the scene similarity metric should be monotonic, in the
    sense that adding correct clipart should always increase the score, adding
    incorrect clipart should decrease it, and removing incorrect clipart should
    increase it. It also breaks out the different components of Mike/Jenny:
    flip, expression, and pose; as well as capping distance error at 1.
    """
    idx1 = set(x.idx for x in target)
    idx2 = set(x.idx for x in pred)
    iou = len(idx1 & idx2) / len(idx1 | idx2)

    intersection_size = len(idx1 & idx2)
    union_size = len(idx1 | idx2)

    common_idxs = list(idx1 & idx2)
    match1 = [[x for x in target if x.idx == idx][0] for idx in common_idxs]
    match2 = [[x for x in pred if x.idx == idx][0] for idx in common_idxs]

    num = np.zeros(8)
    denom = np.zeros(8)

    num[0] = intersection_size

    for c1, c2 in zip(match1, match2):
        num[1] += int(c1.flip != c2.flip)
        if c1.idx in c1.HUMAN_IDXS:
            num[2] += int(c1.expression != c2.expression)
            num[3] += int(c1.pose != c2.pose)
        num[4] += int(c1.depth != c2.depth)
        num[5] += min(1.0, np.sqrt((c1.normed_x - c2.normed_x) ** 2 + (c1.normed_y - c2.normed_y) ** 2))

    denom[:6] = union_size

    for idx_i in range(len(match1)):
        for idx_j in range(idx_i, len(match1)):
            if idx_i == idx_j:
                continue
            c1i, c1j = match1[idx_i], match1[idx_j]
            c2i, c2j = match2[idx_i], match2[idx_j]

            # TODO(nikita): this doesn't correctly handle the case if two
            # cliparts have *exactly* the same x/y coordinates in the target
            num[6] += int((c1i.x - c1j.x) * (c2i.x - c2j.x) <= 0)
            num[7] += int((c1i.y - c1j.y) * (c2i.y - c2j.y) <= 0)

    denom[6:] = union_size * (intersection_size - 1)

    denom = np.maximum(denom, 1)

    score_components = num / denom
    score_weights = np.array([5,-1,-0.5,-0.5,-1,-1,-1,-1])

    return score_components @ score_weights

def clipart_similarity_v1(a, b):
    """
    DEPRECATED: use clipart_similarity instead!

    The original clipart similarity metric, before subtype was split into
    pose/expression
    """
    if a.idx != b.idx:
        return 0

    score = 5
    score -= int(a.subtype != b.subtype or a.flip != b.flip)
    score -= int(a.depth != b.depth)
    score -= np.sqrt((a.normed_x - b.normed_x) ** 2 + (a.normed_y - b.normed_y) ** 2)
    return score

def clipart_similarity(a, b):
    """
    This version of the metric splits out subtype into pose/expression, and caps
    distance error at 1.
    """
    if a.idx != b.idx:
        return 0

    score = 5
    score -= int(a.flip != b.flip)
    score -= 0.5 * int(a.expression != b.expression)
    score -= 0.5 * int(a.pose != b.pose)
    score -= int(a.depth != b.depth)
    score -= min(1.0, np.sqrt((a.normed_x - b.normed_x) ** 2 + (a.normed_y - b.normed_y) ** 2))
    return score
