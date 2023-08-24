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


class Var(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        input, dim, correction = args
        return Var.__forward(input, dim, correction)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, dim, correction = inputs
        ctx.save_for_backward(input)
        ctx.dim = dim
        ctx.correction = correction

    @staticmethod
    def backward(ctx, *grad_outputs):
        (input,) = ctx.saved_tensors
        (grad_output,) = grad_outputs
        return Var.__backward(grad_output, input, ctx.dim, ctx.correction)

    @staticmethod
    def __forward(input: torch.Tensor, dim: int, correction: int):
        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        y_size, x_size, y_stride, x_stride = util.size_and_stride(input, dim)

        def grid(meta):
            return (y_size,)

        output = torch.empty(y_size, **factory_kwargs)

        kernel.Var.forward[grid](
            output,
            input,
            y_size,
            x_size,
            y_stride,
            x_stride,
            correction,
            util.dtype(input.dtype),
        )

        return output

    @staticmethod
    def __backward(grad_output: torch.Tensor, input: torch.Tensor, dim: int, correction: int):
        y_size, x_size, y_stride, x_stride = util.size_and_stride(input, dim)

        def grid(meta):
            return (y_size * triton.cdiv(x_size, meta["x_block_size"]),)

        grad_input = torch.zeros_like(input)

        kernel.Var.backward[grid](
            grad_input,
            grad_output,
            input,
            y_size,
            x_size,
            y_stride,
            x_stride,
            correction,
            util.dtype(grad_input.dtype),
        )

        return grad_input, None, None
