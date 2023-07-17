# Copyright 2023 ⓒ Kakao Brain Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton

from trident import math


def fill(x, v):
    with torch.no_grad():
        return x.fill_(v)


def map_dtype(dtype):
    if dtype == torch.float32:
        return triton.language.float32
    if dtype == torch.float16:
        return triton.language.float16
    else:
        raise NotImplementedError(dtype)


def shared_memory_size_per_block():
    return 64 * 1024


def block_size(num_elem, elem_sz):
    return min(
        triton.next_power_of_2(num_elem),
        shared_memory_size_per_block() // elem_sz,
    )


def num_warps(num_elem, elem_sz, corr=1):
    shm_sz = shared_memory_size_per_block()
    blk_sz = block_size(num_elem, elem_sz)
    blk_byte_sz = blk_sz * elem_sz

    return math.clamp(math.prev_pow2(shm_sz // blk_byte_sz) * corr, 4, 32)


def zero(x):
    with torch.no_grad():
        return x.zero_()
