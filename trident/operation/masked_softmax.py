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


class MaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, mask, dim = args

        util.push_trace("MaskedSoftmax.__forward")
        output = MaskedSoftmax.__forward(input, mask, dim)
        util.pop_trace()

        ctx.save_for_backward(output)
        ctx.dim = dim

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        (grad_output,) = grad_outputs
        (output,) = ctx.saved_tensors

        util.push_trace("MaskedSoftmax.__backward")
        grad_input = MaskedSoftmax.__backward(grad_output, output, ctx.dim)
        util.pop_trace()

        return grad_input, None, None

    @staticmethod
    def __forward(input: torch.Tensor, mask: torch.Tensor, dim: torch.int32):
        y_size, x_size, y_stride, x_stride = util.size_and_stride(input, dim)
        output = torch.empty_like(input)

        def grid(meta):
            return (y_size,)

        util.push_trace("kernel.MaskedSoftmax.forward")
        kernel.MaskedSoftmax.forward[grid](
            output,
            input,
            y_size,
            x_size,
            y_stride,
            x_stride,
            mask,
            util.dtype(output.dtype),
            triton.next_power_of_2(x_size),
        )
        util.pop_trace()

        return output

    @staticmethod
    def __backward(grad_output: torch.Tensor, output: torch.Tensor, dim: torch.int32):
        factory_kwargs = {"device": output.device, "dtype": output.dtype}
        y_size, x_size, y_stride, x_stride = util.size_and_stride(output, dim)
        delta = torch.empty(y_size, **factory_kwargs)
        grad_input = torch.empty_like(output)

        def grid(meta):
            return (y_size,)

        util.push_trace("kernel.MaskedSoftmax.backward_delta")
        kernel.MaskedSoftmax.backward_delta[grid](
            delta,
            grad_output,
            output,
            y_size,
            x_size,
            y_stride,
            x_stride,
            util.dtype(delta.dtype),
            triton.next_power_of_2(x_size),
        )
        util.pop_trace()

        def grid(meta):
            return (y_size,)

        util.push_trace("kernel.MaskedSoftmax.backward")
        kernel.MaskedSoftmax.backward[grid](
            grad_input,
            grad_output,
            output,
            delta,
            y_size,
            x_size,
            y_stride,
            x_stride,
            util.dtype(output.dtype),
            triton.next_power_of_2(x_size),
        )
        util.pop_trace()

        return grad_input
