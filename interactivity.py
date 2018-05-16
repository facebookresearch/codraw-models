# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

try:
    get_ipython()
    INTERACTIVE=True
except:
    INTERACTIVE=False

def try_magic(*args, **kwargs):
    if not INTERACTIVE:
        return
    return get_ipython().magic(*args, **kwargs)

def try_cd(loc):
    if not INTERACTIVE:
        return
    return get_ipython().magic(f'%cd {loc}')
