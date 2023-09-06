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

from trident import kernel, util


class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, dim = args
        output = Softmax.__forward(input, dim)

        ctx.save_for_backward(output)
        ctx.dim = dim

        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_output,) = grad_outputs
        (output,) = ctx.saved_tensors
        return Softmax.__backward(grad_output, output, ctx.dim)

    @staticmethod
    def __forward(input: torch.Tensor, dim: int):
        y_size, x_size = input.shape
        output = torch.empty_like(input)

        def grid(meta):
            return (x_size if dim == 0 else y_size,)

        kernel.Softmax.forward[grid](
            output,
            input,
            y_size,
            x_size,
            dim,
            dtype=util.dtype(input.dtype),
        )

        return output

    @staticmethod
    def __backward(grad_output: torch.Tensor, output: torch.Tensor, dim: int):
        factory_kwargs = {"device": grad_output.device}
        y_size, x_size, y_stride, x_stride = util.size_and_stride(output, dim)

        def grid(meta):
            return (y_size,)

        delta = torch.empty(x_size, **factory_kwargs)

        kernel.Softmax.backward_delta[grid](
            delta,
            grad_output,
            output,
            x_size,
            y_size,
            x_stride,
            y_stride,
        )

        grad_input = torch.empty_like(output)

        kernel.Softmax.backward[grid](
            grad_input,
            grad_output,
            output,
            delta,
            x_size,
            y_size,
            x_stride,
            y_stride,
            dtype=util.dtype(output.dtype),
        )

        return grad_input, None
