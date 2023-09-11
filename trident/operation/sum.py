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


class Sum(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, dim = args

        util.push_trace("Sum.__forward")
        output = Sum.__forward(input, dim)
        util.pop_trace()

        ctx.save_for_backward(input)
        ctx.dim = dim

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        (input,) = ctx.saved_tensors
        (grad_output,) = grad_outputs

        util.push_trace("Sum.__backward")
        grad_input = Sum.__backward(grad_output, input, ctx.dim)
        util.pop_trace()

        return grad_input, None

    @staticmethod
    def __forward(input, dim):
        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        y_size, x_size, y_stride, x_stride = util.size_and_stride(input, dim)
        output = torch.empty(y_size, **factory_kwargs)

        def grid(meta):
            return (y_size,)

        util.push_trace("kernel.Sum.forward")
        kernel.Sum.forward[grid](
            output,
            input,
            y_size,
            x_size,
            y_stride,
            x_stride,
            util.dtype(input.dtype),
        )
        util.pop_trace()

        return output

    @staticmethod
    def __backward(grad_output, input, dim):
        y_size, x_size, y_stride, x_stride = util.size_and_stride(input, dim)
        grad_input = torch.zeros_like(input)

        def grid(meta):
            return (y_size,)

        util.push_trace("kernel.Sum.backward")
        kernel.Sum.backward[grid](
            grad_input,
            grad_output,
            y_size,
            x_size,
            y_stride,
            x_stride,
        )
        util.pop_trace()

        return grad_input
