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

from trident import kernel


class Dropout(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        return Dropout.__forward(*args, **kwargs)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_out,) = grad_outputs
        (out,) = ctx.saved_tensors
        return Dropout.__backward(grad_out, out)

    @staticmethod
    def __forward(inp, p):
        inp_sz = inp.numel()

        def grid(meta):
            return (triton.cdiv(inp_sz, meta["inp_bs"]),)

        out = torch.empty_like(inp)
        kernel.Dropout.forward[grid](
            inp, inp_sz, p, torch.random.seed(), out, inp_bs=min(triton.next_power_of_2(inp_sz), 1024)
        )
        return out

    @staticmethod
    def __backward(grad_out, out):
        out_sz = out.numel()

        def grid(meta):
            return (triton.cdiv(out_sz, meta["out_bs"]),)

        grad_inp = torch.empty_like(out)
        kernel.Dropout.backward[grid](grad_out, out, out_sz, grad_inp, out_bs=min(triton.next_power_of_2(out_sz), 1024))
        return grad_inp, None, None
