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


class ShiftGELU(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        (input, bias) = args
        return ShiftGELU.__forward(input, bias)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, bias = inputs
        _, shift = output
        ctx.save_for_backward(input, shift)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, _ = grad_outputs
        input, shift = ctx.saved_tensors
        return ShiftGELU.__backward(grad_output, input, shift)

    @staticmethod
    def __forward(input: torch.Tensor, bias: torch.Tensor):
        y_size, x_size = input.shape
        output = torch.empty_like(input)
        shift = torch.empty_like(input)

        def grid(meta):
            return (y_size * triton.cdiv(x_size, meta["x_block_size"]),)

        kernel.ShiftGELU.forward[grid](output, shift, input, y_size, x_size, bias, util.dtype(input.dtype))

        return output, shift

    @staticmethod
    def __backward(grad_output: torch.Tensor, input: torch.Tensor, shift: torch.Tensor):
        y_size, x_size = input.shape
        grad_input = torch.empty_like(input)

        def grid(meta):
            return (y_size * triton.cdiv(x_size, meta["x_block_size"]),)

        kernel.ShiftGELU.backward[grid](grad_input, grad_output, shift, y_size, x_size, util.dtype(input.dtype))

        return grad_input, torch.sum(grad_input, 0)
