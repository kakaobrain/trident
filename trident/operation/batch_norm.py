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


class BatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, running_mean, running_var, weight, bias, momentum, eps = args

        util.push_trace("BatchNorm.__forward")
        output, mean, var = BatchNorm.__forward(input, running_mean, running_var, weight, bias, momentum, eps)
        util.pop_trace()

        ctx.save_for_backward(input, weight, bias, mean, var)
        ctx.eps = eps

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        util.push_trace("BatchNorm.__backward")
        grad_input, grad_weight, grad_bias = BatchNorm.__backward(*grad_outputs, *ctx.saved_tensors, ctx.eps)
        util.pop_trace()

        return grad_input, None, None, grad_weight, grad_bias, None, None

    @staticmethod
    def __forward(
        input: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        momentum: torch.float32,
        eps: torch.float32,
    ):
        num_batches, y_size, x_size = input.shape
        factory_kwargs = {"device": input.device, "dtype": input.dtype}

        def grid(meta):
            return (y_size,)

        output = torch.empty_like(input)
        mean = torch.empty(y_size, **factory_kwargs)
        var = torch.empty(y_size, **factory_kwargs)

        util.push_trace("kernel.BatchNorm.forward")
        kernel.BatchNorm.forward[grid](
            output,
            mean,
            var,
            input,
            num_batches,
            y_size,
            x_size,
            weight,
            bias,
            running_mean,
            running_var,
            momentum,
            eps,
            util.dtype(input.dtype),
            batch_block_size=triton.next_power_of_2(num_batches),
            x_block_size=triton.next_power_of_2(x_size),
        )
        util.pop_trace()

        return output, mean, var

    @staticmethod
    def __backward(
        grad_output: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        mean: torch.Tensor,
        var: torch.Tensor,
        eps: torch.float32,
    ):
        num_batches, y_size, x_size = input.shape

        def grid(meta):
            return (y_size,)

        grad_input = torch.empty_like(input)
        grad_weight = torch.empty_like(weight) if weight is not None else None
        grad_bias = torch.empty_like(bias) if bias is not None else None

        util.push_trace("kernel.BatchNorm.backward")
        kernel.BatchNorm.backward[grid](
            grad_input,
            grad_weight,
            grad_bias,
            grad_output,
            input,
            weight,
            mean,
            var,
            num_batches,
            y_size,
            x_size,
            eps,
            util.dtype(input.dtype),
            batch_block_size=triton.next_power_of_2(num_batches),
            x_block_size=triton.next_power_of_2(x_size),
        )
        util.pop_trace()

        return grad_input, grad_weight, grad_bias
