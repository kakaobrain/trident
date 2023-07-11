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

from trident import kernel, math, util


class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        return Softmax.__forward(*args, **kwargs)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return Softmax.__backward(*grad_outputs, *ctx.saved_tensors)

    @staticmethod
    def __forward(inp, axis):
        assert inp.is_contiguous()
        assert axis == 1

        num_vec, vec_sz = inp.shape

        def grid(meta):
            return [num_vec]

        out = torch.empty_like(inp)
        blk_sz = util.get_block_size(vec_sz, inp.element_size())
        num_warps = util.get_num_warps(vec_sz, inp.element_size(), 4)

        kernel.Softmax.forward[grid](inp, vec_sz, out, blk_sz, num_warps=num_warps)

        return out

    @staticmethod
    def __backward(grad_out, out):
        assert grad_out.is_contiguous() and out.is_contiguous()

        num_vec, vec_sz = out.shape

        def grid(meta):
            return [num_vec]

        grad_inp = torch.empty_like(out)
        blk_sz = max(triton.next_power_of_2(vec_sz), 16)

        kernel.Softmax.backward[grid](grad_out, out, vec_sz, grad_inp, blk_sz)

        return grad_inp, None
