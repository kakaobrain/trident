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


def map_dtype(dtype):
    if dtype == torch.float32:
        return triton.language.float32
    if dtype == torch.float16:
        return triton.language.float16
    else:
        raise NotImplementedError(dtype)


def get_shared_memory_size_per_block():
    return 64 * 1024


def get_proper_block_size(num_elem, elem_sz):
    return min(triton.next_power_of_2(num_elem), get_shared_memory_size_per_block() // elem_sz)

