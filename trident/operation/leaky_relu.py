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


class LeakyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, negative_slope = args

        util.push_trace("LeakyReLU.__forward")
        output = LeakyReLU.__forward(input, negative_slope)
        util.pop_trace()

        ctx.save_for_backward(input)
        ctx.negative_slope = negative_slope

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        grad_output = grad_outputs[0]
        (input,) = ctx.saved_tensors

        util.push_trace("LeakyReLU.__backward")
        grad_input = LeakyReLU.__backward(grad_output, input, ctx.negative_slope)
        util.pop_trace()

        return grad_input, None

    @staticmethod
    def __forward(input: torch.Tensor, negative_slope: torch.float32):
        x_size = input.numel()
        output = torch.empty_like(input)

        def grid(meta):
            return (triton.cdiv(x_size, meta["x_block_size"]),)

        util.push_trace("kernel.LeakyReLU.forward")
        kernel.LeakyReLU.forward[grid](output, input, x_size, negative_slope, util.dtype(input.dtype))
        util.pop_trace()

        return output

    @staticmethod
    def __backward(grad_output: torch.Tensor, input: torch.Tensor, negative_slope: torch.float32):
        x_size = input.numel()
        grad_input = torch.empty_like(input)

        def grid(meta):
            return [triton.cdiv(x_size, meta["x_block_size"])]

        util.push_trace("kernel.LeakyReLU.backward")
        kernel.LeakyReLU.backward[grid](
            grad_input, grad_output, input, x_size, negative_slope, util.dtype(grad_input.dtype)
        )
        util.pop_trace()

        return grad_input
