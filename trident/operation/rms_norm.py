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

from trident import function, kernel, util


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        input, p, weight, bias, eps = args
        return RMSNorm.__forward(input, p, weight, bias, eps)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, p, weight, bias, eps = inputs
        output, rms = output
        ctx.save_for_backward(input, rms, weight, bias)
        ctx.p = p
        ctx.eps = eps

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, _ = grad_outputs
        input, rms, weight, bias = ctx.saved_tensors
        return RMSNorm.__backward(grad_output, input, ctx.p, rms, weight, bias, ctx.eps)

    @staticmethod
    def __forward(input: torch.Tensor, p: float, weight: torch.Tensor, bias: torch.Tensor, eps: float):
        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        y_size, x_size, y_stride, x_stride = util.size_and_stride(input, 1)
        output = torch.empty_like(input)
        rms = torch.empty(y_size, **factory_kwargs)

        def grid(meta):
            return (y_size,)

        kernel.RMSNorm.forward[grid](
            output,
            rms,
            input,
            y_size,
            x_size,
            y_stride,
            x_stride,
            x_size if p < 0.0 or p > 1.0 else x_size * p,
            weight,
            bias,
            eps,
            util.dtype(input.dtype),
            triton.next_power_of_2(x_size),
        )

        return output, rms

    @staticmethod
    def __backward(
        grad_output: torch.Tensor,
        input: torch.Tensor,
        p: float,
        rms: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ):
        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        y_size, x_size, y_stride, x_stride = util.size_and_stride(input, 1)
        grad_input = torch.empty_like(grad_output)
        grad_weight_staging = torch.empty((y_size, x_size), **factory_kwargs)

        def grid(meta):
            return (y_size,)

        kernel.RMSNorm.backward[grid](
            grad_input,
            grad_weight_staging,
            grad_output,
            input,
            y_size,
            x_size,
            y_stride,
            x_stride,
            rms,
            x_size if p < 0.0 or p > 1.0 else x_size * p,
            weight,
            eps,
            util.dtype(input.dtype),
            triton.next_power_of_2(x_size),
        )

        grad_weight = function.sum(grad_weight_staging, 0)

        if bias is not None:
            grad_bias = function.sum(grad_output, 0)
        else:
            grad_bias = None

        return grad_input, None, grad_weight, grad_bias, None
