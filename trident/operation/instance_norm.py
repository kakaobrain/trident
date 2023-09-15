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


class InstanceNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps = args

        util.push_trace("InstanceNorm.__forward")
        output, mean, var = InstanceNorm.__forward(
            input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps
        )
        util.pop_trace()

        ctx.save_for_backward(input, running_mean, running_var, weight, mean, var, weight, bias)
        ctx.use_input_stats = use_input_stats
        ctx.eps = eps

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        input, running_mean, running_var, weight, mean, var, weight, bias = ctx.saved_tensors
        (grad_output,) = grad_outputs

        util.push_trace("InstanceNorm.__backward")
        grad_input, grad_weight, grad_bias = InstanceNorm.__backward(
            grad_output, input, running_mean, running_var, mean, var, weight, bias, ctx.use_input_stats, ctx.eps
        )
        util.pop_trace()

        return grad_input, None, None, grad_weight, grad_bias, None, None, None, None

    @staticmethod
    def __forward(
        input: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        use_input_stats: torch.bool,
        momentum: torch.float32,
        eps: torch.float32,
    ):
        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        num_batches, y_size, x_size = input.shape
        output = torch.empty_like(input)
        mean = torch.empty(num_batches, y_size, **factory_kwargs)
        var = torch.empty(num_batches, y_size, **factory_kwargs)

        def grid(meta):
            return (num_batches * y_size,)

        util.push_trace("kernel.InstanceNorm.forward")
        kernel.InstanceNorm.forward[grid](
            output,
            mean,
            var,
            input,
            y_size,
            x_size,
            running_mean,
            running_var,
            weight,
            bias,
            1 if use_input_stats else 0,
            eps,
            util.dtype(input.dtype),
            triton.next_power_of_2(x_size),
        )
        util.pop_trace()

        if use_input_stats and running_mean is not None and running_var is not None:

            def grid(meta):
                return (y_size,)

            util.push_trace("kernel.InstanceNorm.forward_running_mean_running_var")
            kernel.InstanceNorm.forward_running_mean_running_var[grid](
                mean,
                var,
                running_mean,
                running_var,
                num_batches,
                y_size,
                x_size,
                momentum,
                triton.next_power_of_2(num_batches),
            )
            util.pop_trace()

        return output, mean, var

    @staticmethod
    def __backward(
        grad_output: torch.Tensor,
        input: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        mean: torch.Tensor,
        var: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        use_input_stats: torch.bool,
        eps: torch.float,
    ):
        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        num_batches, y_size, x_size = input.shape
        grad_input = torch.empty_like(input)

        if weight is not None:
            grad_weight_staging = torch.zeros(num_batches, y_size, **factory_kwargs)
        else:
            grad_weight_staging = None

        if bias is not None:
            grad_bias_staging = torch.zeros(num_batches, y_size, **factory_kwargs)
        else:
            grad_bias_staging = None

        def grid(meta):
            return (num_batches * y_size,)

        util.push_trace("kernel.InstanceNorm.backward")
        kernel.InstanceNorm.backward[grid](
            grad_input,
            grad_weight_staging,
            grad_bias_staging,
            grad_output,
            input,
            y_size,
            x_size,
            running_mean,
            running_var,
            mean,
            var,
            weight,
            1 if use_input_stats else 0,
            eps,
            util.dtype(grad_input.dtype),
            triton.next_power_of_2(x_size),
        )
        util.pop_trace()

        if grad_weight_staging is not None:
            util.push_trace("torch.sum")
            grad_weight = torch.sum(grad_weight_staging, 0)
            util.pop_trace()
        else:
            grad_weight = None

        if grad_bias_staging is not None:
            util.push_trace("torch.sum")
            grad_bias = torch.sum(grad_bias_staging, 0)
            util.pop_trace()
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias
