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

from trident import math, module


def fill(inp, val):
    with torch.no_grad():
        return inp.fill_(val)


def dtype(inp):
    if inp == torch.float32:
        return triton.language.float32
    if inp == torch.float16:
        return triton.language.float16
    else:
        raise NotImplementedError(inp)


def shared_memory_size_per_block():
    return 64 * 1024


def block_size(num_elem, elem_sz):
    return min(
        triton.next_power_of_2(num_elem),
        shared_memory_size_per_block() // elem_sz,
    )


def optimize_module(mod):
    opt_mod = None

    if isinstance(mod, torch.nn.Dropout):
        opt_mod = module.Dropout(mod.p)
    elif isinstance(mod, torch.nn.GroupNorm):
        opt_mod = module.GroupNorm(
            mod.num_groups, mod.num_channels, mod.eps, mod.affine
        )
    elif isinstance(mod, torch.nn.InstanceNorm1d):
        opt_mod = module.InstanceNorm1d(
            mod.num_features, mod.eps, mod.momentum, mod.affine, mod.track_running_stats
        )
    elif isinstance(mod, torch.nn.InstanceNorm2d):
        opt_mod = module.InstanceNorm1d(
            mod.num_features, mod.eps, mod.momentum, mod.affine, mod.track_running_stats
        )
    elif isinstance(mod, torch.nn.LayerNorm):
        opt_mod = module.LayerNorm(
            mod.normalized_shape, mod.eps, mod.elementwise_affine
        )
    elif isinstance(mod, torch.nn.Softmax):
        opt_mod = module.Softmax(mod.dim)

    if opt_mod is not None:
        opt_mod.load_state_dict(mod.state_dict())

    return opt_mod


def optimize_model(model):
    for name, child in model.named_children():
        if other := optimize_module(child):
            setattr(model, name, other)

        optimize_model(child)


def num_warps(num_elem, elem_sz, corr=1):
    shm_sz = shared_memory_size_per_block()
    blk_sz = block_size(num_elem, elem_sz)
    blk_byte_sz = blk_sz * elem_sz

    return math.clamp(math.prev_pow2(shm_sz // blk_byte_sz) * corr, 4, 32)


def zero(inp):
    with torch.no_grad():
        return inp.zero_()
