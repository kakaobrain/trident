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

        util.push_trace("PReLU.__forward")
        output = PReLU.__forward(input.view(PReLU.__shape(input)), weight)
        util.pop_trace()

        ctx.save_for_backward(input, weight)

        return output.view(input.shape)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        grad_output = grad_outputs[0]
        input, weight = ctx.saved_tensors

        util.push_trace("PReLU.__backward")
        grad_input, grad_weight = PReLU.__backward(grad_output, input.view(PReLU.__shape(input)), weight)
        util.pop_trace()

        return grad_input.view(input.shape), grad_weight, None

    @staticmethod
    def __forward(input: torch.Tensor, weight: torch.Tensor):
        num_batches, y_size, x_size = input.shape
        output = torch.empty_like(input)

        def grid(meta):
            num_y_blocks = triton.cdiv(y_size, meta["y_block_size"])
            num_x_blocks = triton.cdiv(x_size, meta["x_block_size"])
            return (num_batches * num_y_blocks * num_x_blocks,)

        util.push_trace("kernel.PReLU.forward")
        kernel.PReLU.forward[grid](
            output,
            input,
            weight,
            num_batches,
            y_size,
            x_size,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            util.dtype(output.dtype),
        )
        util.pop_trace()

        return output

    @staticmethod
    def __backward(grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor):
        num_batches, y_size, x_size = input.shape
        grad_input = torch.empty_like(input)
        grad_weight_staging = torch.empty_like(input)

        def grid(meta):
            num_y_blocks = triton.cdiv(y_size, meta["y_block_size"])
            num_x_blocks = triton.cdiv(x_size, meta["x_block_size"])
            return (num_batches * num_y_blocks * num_x_blocks,)

        util.push_trace("kernel.PReLU.backward")
        kernel.PReLU.backward[grid](
            grad_input,
            grad_weight_staging,
            grad_output,
            input,
            weight,
            num_batches,
            y_size,
            x_size,
            grad_input.stride(0),
            grad_input.stride(1),
            grad_input.stride(2),
        )
        util.pop_trace()

        if grad_weight_staging.dim() < 3:
            grad_weight = grad_weight_staging
        else:
            util.push_trace("torch.sum")
            grad_weight = torch.sum(grad_weight_staging, 2)
            util.pop_trace()

        return grad_input, grad_weight

    @staticmethod
    def __shape(input: torch.Tensor):
        if input.dim() == 1:
            return 1, 1, -1
        elif input.dim() == 2:
            return *input.shape, 1
        else:
            return *input.shape[0:2], -1
