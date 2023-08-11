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


class GroupNorm(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        input, num_groups, weight, bias, eps = args
        return GroupNorm.__forward(input, num_groups, weight, bias, eps)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, num_groups, weight, bias, eps = inputs
        ctx.save_for_backward(input, weight, bias)
        ctx.num_groups = num_groups
        ctx.eps = eps

    @staticmethod
    def backward(ctx, *grad_outputs):
        return GroupNorm.__backward(*grad_outputs, *ctx.saved_tensors, ctx.num_groups, ctx.eps)

    @staticmethod
    def __forward(input, num_groups, weight, bias, eps):
        num_batches, y_size, x_size = input.shape
        output = torch.zeros_like(input)

        def grid(meta):
            return (num_batches * num_groups,)

        kernel.GroupNorm.forward[grid](
            output,
            input,
            y_size,
            x_size,
            num_groups,
            weight,
            bias,
            eps,
            group_block_size=triton.next_power_of_2(y_size // num_groups),
            dtype=util.dtype(input.dtype),
        )
        return output.view(num_batches, y_size, x_size)

    @staticmethod
    def __backward(grad_output, inp, weight, bias, num_groups, eps):
        factory_kwargs = {"device": grad_output.device, "dtype": grad_output.dtype}
        num_batches, y_size, x_size = inp.shape
        grad_input = torch.empty_like(inp)

        if weight is not None:
            grad_weight_staging = torch.empty((num_batches, y_size), **factory_kwargs)
        else:
            grad_weight_staging = None

        if bias is not None:
            grad_bias_staging = torch.empty((num_batches, y_size), **factory_kwargs)
        else:
            grad_bias_staging = None

        def grid(meta):
            return (num_batches * num_groups,)

        kernel.GroupNorm.backward[grid](
            grad_input,
            grad_weight_staging,
            grad_bias_staging,
            grad_output,
            inp,
            y_size,
            x_size,
            num_groups,
            weight,
            eps,
            triton.next_power_of_2(x_size),
            triton.next_power_of_2(y_size // num_groups),
            util.dtype(grad_output.dtype),
            num_warps=4,
        )

        if weight is not None:
            grad_weight = function.sum(grad_weight_staging, 0)
        else:
            grad_weight = None

        if bias is not None:
            grad_bias = function.sum(grad_bias_staging, 0)
        else:
            grad_bias = None

        return grad_input, None, grad_weight, grad_bias, None, None, None, None
