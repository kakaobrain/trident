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

import torch
import triton

from trident import kernel, util


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        input, normalized_shape, weight, bias, eps = args
        return LayerNorm.__forward(input, normalized_shape, weight, bias, eps)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, normalized_shape, weight, bias, eps = inputs
        output, rstd, mean = output
        ctx.save_for_backward(input, weight, bias, rstd, mean)
        ctx.normalized_shape = normalized_shape

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, _, _ = grad_outputs
        input, weight, bias, rstd, mean = ctx.saved_tensors
        return LayerNorm.__backward(grad_output, input, ctx.normalized_shape, weight, bias, rstd, mean)

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

        return output, rstd, mean

    @staticmethod
    def __backward(grad_output, input, normalized_shape, weight, bias, rstd, mean):
        factory_kwargs = {"device": grad_output.device, "dtype": grad_output.dtype}
        x_size = functools.reduce(lambda x, y: x * y, normalized_shape)
        y_size = grad_output.numel() // x_size
        grad_input = torch.empty_like(input)

        if weight is not None:
            grad_weight_staging = torch.empty((y_size, x_size), **factory_kwargs)
            # grad_weight_staging = torch.empty((x_size, y_size), **factory_kwargs)
        else:
            grad_weight_staging = None

        def grid(meta):
            return (y_size,)

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

        if weight is not None:
            grad_weight = torch.sum(grad_weight_staging, 0)
        else:
            grad_weight = None

        if bias is not None:
            grad_bias = torch.sum(grad_output, 0)
        else:
            grad_bias = None

        return grad_input, None, grad_weight, grad_bias, None, None, None, None
