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


class SiLU(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        return SiLU.__forward(*args, **kwargs)

    @staticmethod
    def setup_context(ctx, inputs, output):
        inp, = inputs
        ctx.save_for_backward(inp)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return SiLU.__backward(*grad_outputs, *ctx.saved_tensors)

    @staticmethod
    def __forward(inp):
        out = torch.empty_like(inp)
        inp_sz = inp.numel()

        def grid(meta):
            return (triton.cdiv(inp_sz, meta['inp_bs']),)


        inp_bs = min(triton.next_power_of_2(inp_sz), 1024)
        dtype = inp.dtype

        kernel.SiLU.forward[grid](inp, inp_sz, out,
                                  inp_bs=inp_bs, dtype=dtype)

        return out

    @staticmethod
    def __backward(grad_out, inp):
        grad_inp = torch.empty_like(inp)
        inp_sz = inp.numel()

        def grid(meta):
            return (triton.cdiv(inp_sz, meta['inp_bs']),)

        inp_bs = min(triton.next_power_of_2(inp_sz), 1024)
        dtype = inp.dtype

        kernel.SiLU.backward[grid](grad_out, inp, inp_sz, grad_inp,
                                   inp_bs=inp_bs, dtype=dtype)

        return grad_inp, None
