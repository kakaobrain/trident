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


class SiLU(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        return SiLU.__forward(*args, **kwargs)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (input,) = inputs
        ctx.save_for_backward(input)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_output,) = grad_outputs
        (input,) = ctx.saved_tensors
        return SiLU.__backward(grad_output, input)

    @staticmethod
    def __forward(input: torch.Tensor):
        output = torch.empty_like(input)

        def grid(meta):
            return (triton.cdiv(input.numel(), meta["x_block_size"]),)

        kernel.SiLU.forward[grid](output, input, input.numel(), util.dtype(input.dtype))

        return output

    @staticmethod
    def __backward(grad_output: torch.Tensor, input: torch.Tensor):
        grad_input = torch.empty_like(input)

        def grid(meta):
            return (triton.cdiv(input.numel(), meta["x_block_size"]),)

        kernel.SiLU.backward[grid](grad_input, grad_output, input, input.numel(), util.dtype(input.dtype))

        return grad_input, None
