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


class CosineSimilarity(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        x1, x2, dim, eps = args

        util.push_trace("CosineSimilarity.__forward")
        output, denominator, numerator = CosineSimilarity.__forward(x1, x2, dim, eps)
        util.pop_trace()

        ctx.save_for_backward(x1, x2, denominator, numerator)
        ctx.dim = dim

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        grad_output = grad_outputs[0]
        x1, x2, denominator, numerator = ctx.saved_tensors

        util.push_trace("CosineSimilarity.__backward")
        grad_x1, grad_x2 = CosineSimilarity.__backward(grad_output, x1, x2, denominator, numerator, ctx.dim)
        util.pop_trace()

        return grad_x1, grad_x2, None, None

    @staticmethod
    def __forward(x1: torch.Tensor, x2: torch.Tensor, dim: torch.int32, eps: torch.float32):
        assert x1.is_contiguous() and x2.is_contiguous() and x1.shape == x2.shape

        factory_kwargs = {"device": x1.device, "dtype": x1.dtype}
        z_size, y_size, x_size, z_stride, y_stride, x_stride = util.size_and_stride(x1, dim)
        output_y_size, output_x_size, size_along_dim = CosineSimilarity.__output_size_and_size_along_dim(x1, dim)
        output = torch.empty(output_y_size, output_x_size, **factory_kwargs)
        denominator = torch.empty_like(output)
        numerator = torch.empty_like(output)

        def grid(meta):
            return (output_y_size * output_x_size,)

        util.push_trace("kernel.CosineSimilarity.forward")
        kernel.CosineSimilarity.forward[grid](
            output,
            denominator,
            numerator,
            x1,
            x2,
            z_size,
            y_size,
            x_size,
            z_stride,
            y_stride,
            x_stride,
            eps,
            size_along_dim,
            output_y_size,
            output_x_size,
            util.dtype(x1.dtype),
        )
        util.pop_trace()

        return output, denominator, numerator

    @staticmethod
    def __backward(grad_output, x1, x2, denominator, numerator, dim):
        grad_x1 = torch.empty_like(x1)
        grad_x2 = torch.empty_like(x2)

        z_size, y_size, x_size, z_stride, y_stride, x_stride = util.size_and_stride(x1, dim)
        output_y_size, output_x_size, size_along_dim = CosineSimilarity.__output_size_and_size_along_dim(x1, dim)

        def grid(meta):
            return (output_y_size * output_x_size,)

        util.push_trace("kernel.CosineSimilarity.backward")
        kernel.CosineSimilarity.backward[grid](
            grad_x1,
            grad_x2,
            grad_output,
            denominator,
            numerator,
            x1,
            x2,
            z_size,
            y_size,
            x_size,
            z_stride,
            y_stride,
            x_stride,
            size_along_dim,
            output_y_size,
            output_x_size,
            util.dtype(x1.dtype),
        )
        util.pop_trace()

        return grad_x1, grad_x2

    @staticmethod
    def __output_size_and_size_along_dim(input: torch.Tensor, dim: int):
        z_size, y_size, x_size = input.shape

        if dim == 0:
            output_y_size, output_x_size = y_size, x_size
            size_along_dim = z_size
        elif dim == 1:
            output_y_size, output_x_size = z_size, x_size
            size_along_dim = y_size
        else:
            output_y_size, output_x_size = z_size, y_size
            size_along_dim = x_size

        return output_y_size, output_x_size, size_along_dim
