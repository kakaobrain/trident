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

from trident import kernel, util


class GELU(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        return GELU.__forward(*args, **kwargs)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return GELU.__backward(*grad_outputs, *ctx.saved_tensors)

    @staticmethod
    def __forward(inp):
        inp_sz = inp.numel()

        def grid(meta):
            return [triton.cdiv(inp_sz, meta["blk_sz"])]

        out = torch.empty_like(inp)
        blk_sz = util.block_size(inp_sz, inp.element_size())

        kernel.GELU.forward[grid](inp, inp_sz, out, blk_sz)

        return out

    @staticmethod
    def __backward(grad_out, inp):
        inp_sz = inp.numel()

        def grid(meta):
            return [triton.cdiv(inp_sz, meta["blk_sz"])]

        grad_inp = torch.empty_like(inp)
        blk_sz = util.block_size(inp_sz, inp.element_size())

        kernel.GELU.backward[grid](grad_out, inp, grad_inp, inp_sz, blk_sz)

        return grad_inp
