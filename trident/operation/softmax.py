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

from trident import kernel, math, util


class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        input, dim = args
        return Softmax.__forward(input, dim)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, dim = inputs
        ctx.save_for_backward(output)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_output,) = grad_outputs
        (output,) = ctx.saved_tensors
        return Softmax.__backward(grad_output, output, ctx.dim)

    @staticmethod
    def __forward(input: torch.Tensor, dim: int):
        assert dim == 1

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
        y_size, x_size = output.shape
        grad_input = torch.empty_like(output)
        block_size = max(triton.next_power_of_2(x_size), 16)

        def grid(meta):
            return (x_size if dim == 0 else y_size,)

        kernel.Softmax.backward[grid](grad_output, output, x_size, grad_input, block_size)

        return grad_input, None
