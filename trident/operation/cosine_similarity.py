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

from trident import kernel, util


class CosineSimilarity(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        x1, x2, dim, eps = args
        return CosineSimilarity.__forward(x1, x2, dim, eps)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x1, x2, dim, eps = inputs
        _, denominator, numerator = output
        ctx.save_for_backward(x1, x2, denominator, numerator)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        x1, x2, denominator, numerator = ctx.saved_tensors
        return CosineSimilarity.__backward(grad_output, x1, x2, denominator, numerator, ctx.dim)

    @staticmethod
    def __forward(x1: torch.Tensor, x2: torch.Tensor, dim: torch.int32, eps: torch.float32):
        assert x1.is_contiguous() and x2.is_contiguous() and x1.shape == x2.shape

        factory_kwargs = {"device": x1.device, "dtype": x1.dtype}
        num_batches, y_size, x_size = x1.shape

        if dim == 0:
            grid_size = y_size * x_size
            size_along_dim = num_batches
            output = torch.empty(y_size, x_size, **factory_kwargs)
        elif dim == 1:
            grid_size = num_batches * x_size
            size_along_dim = y_size
            output = torch.empty(num_batches, x_size, **factory_kwargs)
        else:
            grid_size = num_batches * y_size
            size_along_dim = x_size
            output = torch.empty(num_batches, y_size, **factory_kwargs)

        denominator = output.clone()
        numerator = output.clone()

        def grid(meta):
            return (grid_size,)

        kernel.CosineSimilarity.forward[grid](
            output,
            denominator,
            numerator,
            x1,
            x2,
            num_batches,
            y_size,
            x_size,
            eps,
            size_along_dim,
            dim,
            util.dtype(x1.dtype),
        )

        return output, denominator, numerator

    @staticmethod
    def __backward(grad_output, x1, x2, denominator, numerator, dim):
        num_batches, y_size, x_size = x1.shape

        grad_x1 = torch.empty_like(x1)
        grad_x2 = torch.empty_like(x2)

        if dim == 0:
            grid_size = y_size * x_size
            size_along_dim = num_batches
        elif dim == 1:
            grid_size = num_batches * x_size
            size_along_dim = y_size
        else:
            grid_size = num_batches * y_size
            size_along_dim = x_size

        def grid(meta):
            return (grid_size,)

        kernel.CosineSimilarity.backward[grid](
            grad_x1,
            grad_x2,
            grad_output,
            denominator,
            numerator,
            x1,
            x2,
            num_batches,
            y_size,
            x_size,
            size_along_dim,
            dim,
            util.dtype(x1.dtype),
        )

        return grad_x1, grad_x2, None, None
