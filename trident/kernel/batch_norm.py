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

import triton
import triton.language as tl


class BatchNorm:
    @staticmethod
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
        mean_ptr: tl.tensor,
        var_ptr: tl.tensor,
        input_ptr: tl.tensor,
        num_batches: tl.int32,
        y_size: tl.int32,
        x_size: tl.int32,
        weight_ptr: tl.tensor,
        bias_ptr: tl.tensor,
        running_mean_ptr: tl.tensor,
        running_var_ptr: tl.tensor,
        momentum: tl.float32,
        eps: tl.float32,
        dtype: tl.constexpr,
        batch_block_size: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)

        output_block_ptr = tl.make_block_ptr(
            output_ptr + pid * x_size,
            shape=(num_batches, x_size),
            strides=(y_size * x_size, 1),
            offsets=(0, 0),
            block_shape=(batch_block_size, x_block_size),
            order=(1, 0),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr + pid * x_size,
            shape=(num_batches, x_size),
            strides=(y_size * x_size, 1),
            offsets=(0, 0),
            block_shape=(batch_block_size, x_block_size),
            order=(1, 0),
        )

        batch_condition = tl.arange(0, batch_block_size) < num_batches
        x_condition = tl.arange(0, x_block_size) < x_size
        condition = batch_condition[:, None] & x_condition[None, :]
        denominator = num_batches * x_size
        input = tl.load(input_block_ptr, boundary_check=(0, 1), padding_option="zero")
        mean = tl.sum(input / denominator)
        deviation = tl.where(condition, input - mean, 0)
        var = tl.sum(deviation * deviation / denominator)
        std = tl.sqrt(var + eps)
        output = (input - mean) / std

        if weight_ptr is not None:
            weight = tl.load(weight_ptr + pid)
            output = output * weight

        if bias_ptr is not None:
            bias = tl.load(bias_ptr + pid)
            output = output + bias

        tl.store(output_block_ptr, output.to(dtype), boundary_check=(0, 1))
        tl.store(mean_ptr + pid, mean)
        tl.store(var_ptr + pid, var)

        if running_mean_ptr is not None:
            running_mean = tl.load(running_mean_ptr + pid)
            tl.store(running_mean_ptr + pid, running_mean * (1 - momentum) + mean * momentum)

        if running_var_ptr is not None:
            running_var = tl.load(running_var_ptr + pid)
            tl.store(
                running_var_ptr + pid, running_var * (1 - momentum) + var * (denominator / (denominator - 1)) * momentum
            )

    @staticmethod
    @triton.jit
    def backward(
        grad_input_ptr: tl.tensor,
        grad_weight_ptr: tl.tensor,
        grad_bias_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        weight_ptr: tl.tensor,
        mean_ptr: tl.tensor,
        var_ptr: tl.tensor,
        num_batches: tl.int32,
        y_size: tl.int32,
        x_size: tl.int32,
        eps: tl.float32,
        dtype: tl.constexpr,
        batch_block_size: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)

        grad_input_block_ptr = tl.make_block_ptr(
            grad_input_ptr + pid * x_size,
            shape=(num_batches, x_size),
            strides=(y_size * x_size, 1),
            offsets=(0, 0),
            block_shape=(batch_block_size, x_block_size),
            order=(1, 0),
        )
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr + pid * x_size,
            shape=(num_batches, x_size),
            strides=(y_size * x_size, 1),
            offsets=(0, 0),
            block_shape=(batch_block_size, x_block_size),
            order=(1, 0),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr + pid * x_size,
            shape=(num_batches, x_size),
            strides=(y_size * x_size, 1),
            offsets=(0, 0),
            block_shape=(batch_block_size, x_block_size),
            order=(1, 0),
        )

        batch_condition = tl.arange(0, batch_block_size) < num_batches
        x_condition = tl.arange(0, x_block_size) < x_size
        condition = batch_condition[:, None] & x_condition[None, :]
        denominator = num_batches * x_size
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0, 1), padding_option="zero")
        input = tl.load(input_block_ptr, boundary_check=(0, 1), padding_option="zero")
        weight = tl.load(weight_ptr + pid) if weight_ptr is not None else 1
        mean = tl.load(mean_ptr + pid)
        var = tl.load(var_ptr + pid)
        std = tl.sqrt(var + eps)
        centered_mean = tl.where(condition, input - mean, 0)
        grad_norm = weight * grad_output
        grad_std = -tl.sum((grad_norm * centered_mean) / (std * std))
        grad_var = 0.5 * grad_std / std
        grad_centered_mean = grad_norm / std + (2.0 / denominator) * centered_mean * grad_var
        grad_mean = tl.sum(tl.where(condition, grad_centered_mean, 0.0) / denominator)
        grad_input = grad_centered_mean - grad_mean

        tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check=(0, 1))

        if grad_weight_ptr:
            input_norm = centered_mean / std
            grad_weight = tl.sum(input_norm * grad_output)
            tl.store(grad_weight_ptr + pid, grad_weight.to(dtype))

        if grad_bias_ptr:
            grad_bias = tl.sum(grad_output)
            tl.store(grad_bias_ptr + pid, grad_bias)
