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


class Max(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, dim = args
        output, argmax = Max.__forward(input, dim)

        ctx.save_for_backward(input, output, argmax)
        ctx.dim = dim

        return output, argmax

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        grad_output, grad_argmax = grad_outputs
        input, output, argmax = ctx.saved_tensors
        return Max.__backward(grad_output, input, argmax, ctx.dim)

    @staticmethod
    def __forward(input: torch.Tensor, dim: torch.int32):
        factory_kwargs = {"device": input.device}
        y_size, x_size, y_stride, x_stride = util.size_and_stride(input, dim)
        output = torch.empty(y_size, **factory_kwargs, dtype=input.dtype)
        argmax = torch.empty(y_size, **factory_kwargs, dtype=torch.int64)

        def grid(meta):
            return (y_size,)

        kernel.Max.forward[grid](
            output,
            argmax,
            input,
            y_size,
            x_size,
            y_stride,
            x_stride,
            triton.next_power_of_2(x_size),
        )

        return output, argmax

    @staticmethod
    def __backward(grad_output: torch.Tensor, input: torch.Tensor, argmax: torch.Tensor, dim: torch.int32):
        y_size, x_size, y_stride, x_stride = util.size_and_stride(input, dim)
        grad_input = torch.zeros_like(input)

        def grid(meta):
            return (y_size,)

        kernel.Max.backward[grid](
            grad_input,
            grad_output,
            argmax,
            y_size,
            x_size,
            y_stride,
            x_stride,
            triton.next_power_of_2(x_size),
        )

        return grad_input, None, None
