# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Abstract Scene (abs) utilities copied from the original CoDraw codebase
"""

import math
import torch
from torch.autograd import Variable
import math

class AbsUtil:
	"""AbsUtil ported from AbsUtil.js"""

	# Various variables setting up the appearence of the interface
	CANVAS_WIDTH = 500
	CANVAS_HEIGHT = 400
	NOT_USED = -10000

	numClipArts = 58
	numTypes = 8
	numProps = 6
	numClasses = [58,35,3,2,1,1]
	Null = 0

	def __init__(self, str):
		# Each object type has its own prefix, the ordering of the object types affects the
		# order in which they are rendered. That is the "t" type (toys) will be rendered on top
		# of the "hb0" (boy) category assuming they have the same depth.
		self.prefix = ['s','p','hb0','hb1','a','c','e','t']

		# Total number of clipart for each type
		self.typeTotalCt = [8,10,35,35,6,10,7,15]

		# Total number of clipart to be randomly selected for each type
		# The sum should equal numClipart
		self.typeCt = [3,4,5,5,2,3,2,4]

		self.str = str
		self.obj = self.preprocess(str)

	# Preprocess given CSV into 7Val format, which is
	# 1. clipartIdx integer [0-57]
	# ~~2. clipartType integer [0-7]~~
	# 3. clipartSubType integer [0-34]
	# 4. depth integer [0-2]
	# 5. flip integer [0-1]
	# 6. x-position float [1-500]
	# 7. y-position float [1-400]
	def preprocess(self, str, verbose=False):
		idx = 1;
		val = [];
		if not str or len(str) < 1:
			return None
		results = str.split(',')
		numClipArts = int(results[0])
		for i in range(numClipArts):
			v = list()
			idx = idx + 1  # png filename
			idx = idx + 1  # clip art local index
			_clipArtObjectIdx = int(results[idx]); idx = idx + 1
			_clipArtTypeIdx = int(results[idx]); idx = idx + 1

			# This code was originally used to read the dataset from Python
			_clipArtX = int(round(float(results[idx]))); idx = idx + 1
			_clipArtY = int(round(float(results[idx]))); idx = idx + 1

			# The javascript code, however, used parseInt instead. This has
			# slightly different rounding behavior, which can be recreated by
			# using the following Python code instead:
			# _clipArtX = float(results[idx]); idx = idx + 1
			# _clipArtY = float(results[idx]); idx = idx + 1
			# _clipArtX = int(math.floor(_clipArtX)) if _clipArtX >= 0 else -int(math.floor(-_clipArtX))
			# _clipArtY = int(math.floor(_clipArtY)) if _clipArtY >= 0 else -int(math.floor(-_clipArtY))

			_clipArtZ = int(results[idx]); idx = idx + 1
			_clipArtFlip = int(results[idx]); idx = idx + 1

			if not verbose and (_clipArtX==AbsUtil.NOT_USED or _clipArtY==AbsUtil.NOT_USED):
				continue

			v.append(self.getClipArtIdx(_clipArtObjectIdx, _clipArtTypeIdx))
			# v.append(_clipArtTypeIdx);  # remove this redundant feature
			v.append(_clipArtObjectIdx if (_clipArtTypeIdx==2 or _clipArtTypeIdx==3) else 0)
			v.append(_clipArtZ)
			v.append(_clipArtFlip)
			v.append(_clipArtX)
			v.append(_clipArtY)
			val.append(v)
		return val

	def asTensor(self):
		if None==self.obj:
			return None
		# notice that position (x & y) is rounded as LongTensor
		t = torch.LongTensor(AbsUtil.numClipArts, 6).fill_(AbsUtil.Null)
		# clipartIdx & clipartSubType are starting with 1
		t[:,:2].add_(-1)
		for v in self.obj:
			clipartIdx = v[0]
			t[clipartIdx].copy_(torch.LongTensor(v))
		t[:,:2].add_(1)
		return t

	def __repr__(self):
		return self.obj.__repr__()

	def getClipArtIdx(self, clipArtObjectIdx, clipArtTypeIdx):
		typeTotalPos = [0,8,18,19,20,26,36,43]
		offset = 0 if (clipArtTypeIdx==2 or clipArtTypeIdx==3) else clipArtObjectIdx
		return typeTotalPos[clipArtTypeIdx] + offset

	# Static methods #############################################################

	# Sample clipart from idx(abs_d - abs_b)>0
	# @param abs_b Tensor(bx58x6)
	# @param abs_d Tensor(bx58x6)
	# @output Tensor(bx6)
	# @output Tensor(bx58)
	@staticmethod
	def sample_abs_c(abs_b, abs_d):
		# using Tensors directly
		abs_b = abs_b.data
		abs_d = abs_d.data
		# bx58
		abs_c_mask = (abs_d - abs_b).abs().sum(2)!=0  # updated cliparts
		# bx58x6
		mask = abs_c_mask.unsqueeze(2).expand_as(abs_d)
		# collapsed x 6
		abs_c = abs_d[mask.byte()].view(-1, abs_d.size(-1))
		return abs_c, abs_c_mask

	# Get abs_c mask, if `r_mask` is given, masked over it.
	# @param abs_b (long, bx58x6): latest drawn scene before prev teller's message
	# @param abs_d (long, bx58x6): latest drawn scene before next teller's message
	# @param r_mask (byte, optional, b)
	# #output c_mask (byte, b): batch mask whether drawn scene is changed or not
	@staticmethod
	def get_c_mask(abs_b, abs_d, r_mask=None):
		if Variable==type(r_mask):
			r_mask = r_mask.data
		_, abs_c_mask = AbsUtil.sample_abs_c(abs_b, abs_d)  # _, bx58
		c_mask = abs_c_mask.sum(1).byte()>0
		if r_mask is not None:
			c_mask = c_mask.mul(r_mask)
		return c_mask
