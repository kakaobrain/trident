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

import functools
from typing import Any

import torch
import triton

from trident import kernel, util


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, normalized_shape, weight, bias, eps = args

        util.push_trace("LayerNorm.__forward")
        output, rstd, mean = LayerNorm.__forward(input, normalized_shape, weight, bias, eps)
        util.pop_trace()

        ctx.save_for_backward(input, weight, bias, rstd, mean)
        ctx.normalized_shape = normalized_shape

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        (grad_output,) = grad_outputs
        input, weight, bias, rstd, mean = ctx.saved_tensors

        util.push_trace("LayerNorm.__backward")
        grad_input, grad_weight, grad_bias = LayerNorm.__backward(
            grad_output, input, ctx.normalized_shape, weight, bias, rstd, mean
        )
        util.pop_trace()

        return grad_input, None, grad_weight, grad_bias, None, None, None, None

    @staticmethod
    def __forward(input, normalized_shape, weight, bias, eps):
        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        x_size = functools.reduce(lambda x, y: x * y, normalized_shape)
        y_size = input.numel() // x_size
        output = torch.empty_like(input)
        rstd = torch.empty(y_size, **factory_kwargs)
        mean = torch.empty(y_size, **factory_kwargs)

        def grid(meta):
            return (y_size,)

        util.push_trace("kernel.LayerNorm.forward")
        kernel.LayerNorm.forward[grid](
            output,
            rstd,
            mean,
            input,
            y_size,
            x_size,
            weight,
            bias,
            eps,
            util.dtype(input.dtype),
            triton.next_power_of_2(x_size),
        )
        util.pop_trace()

        return output, rstd, mean

    @staticmethod
    def __backward(grad_output, input, normalized_shape, weight, bias, rstd, mean):
        factory_kwargs = {"device": grad_output.device, "dtype": grad_output.dtype}
        x_size = functools.reduce(lambda x, y: x * y, normalized_shape)
        y_size = grad_output.numel() // x_size
        grad_input = torch.empty_like(input)

        if weight is not None:
            grad_weight_staging = torch.empty(y_size, x_size, **factory_kwargs)
        else:
            grad_weight_staging = None

        def grid(meta):
            return (y_size,)

        util.push_trace("kernel.LayerNorm.backward")
        kernel.LayerNorm.backward[grid](
            grad_input,
            grad_weight_staging,
            grad_output,
            input,
            y_size,
            x_size,
            weight,
            rstd,
            mean,
            util.dtype(grad_output.dtype),
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
            grad_bias = torch.sum(grad_output, 0)
            util.pop_trace()
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias
