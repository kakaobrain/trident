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

from trident import kernel, math


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
            return (num_vec,)

        out = torch.empty_like(inp)
        vec_bs = triton.next_power_of_2(vec_sz)
        num_wraps = math.clamp(triton.next_power_of_2(vec_sz // 512), 2, 32)

        kernel.Softmax.forward[grid](inp, vec_sz, inp.stride(0), out,
                                     vec_bs=vec_bs, dtype=inp.dtype, num_warps=num_wraps)

        return out

    @staticmethod
    def __backward(grad_out, out):
        grad_x = torch.empty_like(out)

        assert grad_x.is_cuda and grad_x.is_contiguous

        for i, s in enumerate(out):
            jacob = torch.diagflat(s) - torch.mm(s.unsqueeze(0).t(), s.unsqueeze(0))
            grad_x[i] = torch.mm(grad_out[i].unsqueeze(0), jacob)

        return grad_x, None
