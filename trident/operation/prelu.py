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


class PReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, weight = args

        ctx.save_for_backward(input, weight)

        return PReLU.__forward(input, weight)

    @staticmethod
    def __forward(input, weight):
        assert input.is_contiguous() and weight.is_contiguous()

        y_size, x_size = input.shape
        output = torch.empty_like(input)

        def grid(meta):
            return (triton.cdiv(y_size, meta["y_block_size"]) * triton.cdiv(x_size, meta["x_block_size"]),)

        kernel.PReLU.forward[grid](
            output,
            input,
            weight,
            y_size,
            x_size,
        )

        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        return PReLU.__backward(grad_outputs[0], *ctx.saved_tensors)

    @staticmethod
    def __backward(grad_output, input, weight):
        assert input.is_contiguous() and weight.is_contiguous()

        y_size, x_size = input.shape

        grad_input = torch.empty_like(input)
        grad_weight = torch.empty_like(input)

        def grid(meta):
            return (triton.cdiv(y_size, meta["y_block_size"]) * triton.cdiv(x_size, meta["x_block_size"]),)

        kernel.PReLU.backward[grid](grad_input, grad_weight, input, weight, grad_output, y_size, x_size)

        return grad_input, grad_weight, None
