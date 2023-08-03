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


class Sum(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        return Sum.__forward(*args, **kwargs)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, axis = inputs
        ctx.save_for_backward(input)
        ctx.axis = axis

    @staticmethod
    def backward(ctx, *grad_outputs):
        (input,) = ctx.saved_tensors
        (grad_output,) = grad_outputs
        return Sum.__backward(grad_output, input, ctx.axis)

    @staticmethod
    def __forward(input, axis):
        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        y_size, x_size = input.shape

        if axis == 0:
            output_size = x_size
            size_along_axis = y_size
        else:
            output_size = y_size
            size_along_axis = x_size

        def grid(meta):
            return (output_size,)

        output = torch.empty(output_size, **factory_kwargs)

        kernel.Sum.forward[grid](
            output,
            input,
            y_size,
            x_size,
            axis,
            util.block_size(size_along_axis, input.element_size()),
            util.dtype(input.dtype),
        )

        return output

    @staticmethod
    def __backward(grad_output, input, axis):
        y_size, x_size = input.shape

        if axis == 0:
            output_size = y_size
            size_along_axis = x_size
        else:
            output_size = x_size
            size_along_axis = y_size

        def grid(meta):
            return (output_size,)

        grad_input = torch.zeros_like(input)

        kernel.Sum.backward[grid](
            grad_input,
            grad_output,
            y_size,
            x_size,
            axis,
            util.block_size(size_along_axis, input.element_size()),
        )

        return grad_input, None
