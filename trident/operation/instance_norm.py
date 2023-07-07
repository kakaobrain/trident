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

import logging
import torch
import triton

from trident import kernel, util


class InstanceNorm(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        return InstanceNorm.__forward(*args, **kwargs)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("The backward of Instance Norm isn't implemented.")

    @staticmethod
    def __forward(inp, eps, dtype):
        assert inp.dim() == 3 and inp.is_cuda and inp.is_contiguous()

        num_batches, num_ch, vec_sz = inp.shape

        def grid(meta):
            return [num_batches * num_ch]

        out = torch.empty_like(inp)
        vec_blk_sz = util.get_proper_block_size(vec_sz, inp.element_size())

        kernel.InstanceNorm.forward[grid](inp, num_ch, vec_sz, eps, out, vec_blk_sz, util.map_dtype(dtype),
                                          num_warps=16)

        return out
