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

from typing import Any

import torch
import triton

from trident import kernel, util


class ShiftGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, bias = args

        util.push_trace("ShiftGELU.__forward")
        output, shift = ShiftGELU.__forward(input, bias)
        util.pop_trace()

        ctx.save_for_backward(input, shift)

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        input, shift = ctx.saved_tensors
        grad_output = grad_outputs[0]

        util.push_trace("ShiftGELU.__backward")
        grad_input, grad_bias = ShiftGELU.__backward(grad_output, input, shift)
        util.pop_trace()

        return grad_input, grad_bias

    @staticmethod
    def __forward(input: torch.Tensor, bias: torch.Tensor):
        y_size, x_size = input.shape
        output = torch.empty_like(input)
        shift = torch.empty_like(input)

        def grid(meta):
            return (y_size * triton.cdiv(x_size, meta["x_block_size"]),)

        util.push_trace("kernel.ShiftGELU.forward")
        kernel.ShiftGELU.forward[grid](output, shift, input, y_size, x_size, bias, util.dtype(input.dtype))
        util.pop_trace()

        return output, shift

    @staticmethod
    def __backward(grad_output: torch.Tensor, input: torch.Tensor, shift: torch.Tensor):
        y_size, x_size = input.shape
        grad_input = torch.empty_like(input)

        def grid(meta):
            return (y_size * triton.cdiv(x_size, meta["x_block_size"]),)

        util.push_trace("kernel.ShiftGELU.backward")
        kernel.ShiftGELU.backward[grid](grad_input, grad_output, shift, y_size, x_size, util.dtype(input.dtype))
        util.pop_trace()

        util.push_trace("torch.sum")
        grad_bias = torch.sum(grad_input, 0)
        util.pop_trace()

        return grad_input, grad_bias
