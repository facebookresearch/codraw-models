# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import redis

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_PASSWORD = 'YOUR PASSWORD HERE'

REDIS_CONNECTION = None

def connect_to_redis():
    global REDIS_CONNECTION
    if REDIS_CONNECTION is None:
        REDIS_CONNECTION = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, db=0)
    return REDIS_CONNECTION
