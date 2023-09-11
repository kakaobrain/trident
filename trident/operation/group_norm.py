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

from trident import function, kernel, util


class GroupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, num_groups, weight, bias, eps = args

        util.push_trace("GroupNorm.__forward")
        output, rstd, mean = GroupNorm.__forward(input, num_groups, weight, bias, eps)
        util.pop_trace()

        ctx.save_for_backward(input, weight, bias, rstd, mean)
        ctx.num_groups = num_groups

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        (grad_output,) = grad_outputs
        input, weight, bias, rstd, mean = ctx.saved_tensors

        util.push_trace("GroupNorm.__backward")
        grad_input, grad_weight, grad_bias = GroupNorm.__backward(
            grad_output, input, weight, bias, rstd, mean, ctx.num_groups
        )
        util.pop_trace()

        return grad_input, None, grad_weight, grad_bias, None, None, None, None

    @staticmethod
    def __forward(
        input: torch.Tensor, num_groups: torch.int, weight: torch.Tensor, bias: torch.Tensor, eps: torch.float
    ):
        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        num_batches, y_size, x_size = input.shape
        output = torch.zeros_like(input)
        rstd = torch.empty((num_batches, num_groups), **factory_kwargs)
        mean = torch.empty((num_batches, num_groups), **factory_kwargs)

        def grid(meta):
            return (num_batches * num_groups,)

        util.push_trace("kernel.GroupNorm.forward")
        kernel.GroupNorm.forward[grid](
            output,
            input,
            rstd,
            mean,
            y_size,
            x_size,
            num_groups,
            weight,
            bias,
            eps,
            util.dtype(input.dtype),
            triton.next_power_of_2(y_size // num_groups),
            triton.next_power_of_2(x_size),
        )
        util.pop_trace()

        return output, rstd, mean

    @staticmethod
    def __backward(
        grad_output: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        rstd: torch.Tensor,
        mean: torch.Tensor,
        num_groups: torch.int,
    ):
        factory_kwargs = {"device": grad_output.device, "dtype": grad_output.dtype}
        num_batches, y_size, x_size = input.shape
        grad_input = torch.empty_like(input)

        if weight is not None:
            grad_weight_staging = torch.empty((num_batches, y_size), **factory_kwargs)
        else:
            grad_weight_staging = None

        if bias is not None:
            grad_bias_staging = torch.empty((num_batches, y_size), **factory_kwargs)
        else:
            grad_bias_staging = None

        def grid(meta):
            return (num_batches * num_groups,)

        util.push_trace("kernel.GroupNorm.backward")
        kernel.GroupNorm.backward[grid](
            grad_input,
            grad_weight_staging,
            grad_bias_staging,
            grad_output,
            input,
            y_size,
            x_size,
            num_groups,
            weight,
            rstd,
            mean,
            util.dtype(grad_output.dtype),
            triton.next_power_of_2(y_size // num_groups),
            triton.next_power_of_2(x_size),
        )
        util.pop_trace()

        if weight is not None:
            util.push_trace("torch.sum")
            grad_weight = torch.sum(grad_weight_staging, 0)
            util.pop_trace()
        else:
            grad_weight = None

        if bias is not None:
            util.push_trace("torch.sum")
            grad_bias = torch.sum(grad_bias_staging, 0)
            util.pop_trace()
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias
