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

from trident import function, kernel, util


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        input, normalized_shape, weight, bias, eps = args
        return LayerNorm.__forward(input, normalized_shape, weight, bias, eps)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, normalized_shape, weight, bias, eps = inputs
        ctx.save_for_backward(input, weight, bias)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_output,) = grad_outputs
        input, weight, bias = ctx.saved_tensors
        return LayerNorm.__backward(grad_output, input, ctx.normalized_shape, weight, bias, ctx.eps)

    @staticmethod
    def __forward(input, normalized_shape, weight, bias, eps):
        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        x_size = functools.reduce(lambda x, y: x * y, normalized_shape)
        y_size = input.numel() // x_size
        output = torch.empty((y_size, x_size), **factory_kwargs)

        def grid(meta):
            return (y_size,)

        kernel.LayerNorm.forward[grid](
            output,
            input,
            y_size,
            x_size,
            weight,
            bias,
            eps,
            block_size=util.block_size(x_size, input.element_size()),
            dtype=util.dtype(input.dtype),
            num_warps=util.num_warps(x_size, input.element_size()),
        )

        return output

    @staticmethod
    def __backward(grad_output, input, normalized_shape, weight, bias, eps):
        factory_kwargs = {"device": grad_output.device, "dtype": grad_output.dtype}
        x_size = functools.reduce(lambda x, y: x * y, normalized_shape)
        y_size = grad_output.numel() // x_size
        grad_input = torch.empty_like(input)

        if weight is not None:
            grad_weight_staging = torch.empty((y_size, x_size), **factory_kwargs)
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
            eps,
            block_size=triton.next_power_of_2(x_size),
            dtype=util.dtype(grad_output.dtype),
        )

        if weight is not None:
            grad_weight = function.sum(grad_weight_staging, 0)
        else:
            grad_weight = None

        if bias is not None:
            grad_bias = function.sum(grad_output, 0)
        else:
            grad_bias = None

        return grad_input, None, grad_weight, grad_bias, None, None, None, None
