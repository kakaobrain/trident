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


class InstanceNorm:
    @staticmethod
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
        mean_ptr: tl.tensor,
        var_ptr: tl.tensor,
        input_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        running_mean_ptr: tl.tensor,
        running_var_ptr: tl.tensor,
        weight_ptr: tl.tensor,
        bias_ptr: tl.tensor,
        use_input_stats: tl.int32,
        eps: tl.float32,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch = pid // y_size
        y_offset = pid % y_size

        output_block_ptr = tl.make_block_ptr(
            output_ptr + batch * y_size * x_size,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr + batch * y_size * x_size,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )

        input = tl.load(input_block_ptr, boundary_check=(1,), padding_option="zero")

        if use_input_stats == 1:
            mean_block_ptr = tl.make_block_ptr(
                mean_ptr + batch * y_size,
                shape=(y_size,),
                strides=(1,),
                offsets=(y_offset,),
                block_shape=(1,),
                order=(0,),
            )
            mean = (tl.sum(input / x_size, 1)).to(dtype)
            tl.store(mean_block_ptr, mean)
        else:
            running_mean_block_ptr = tl.make_block_ptr(
                running_mean_ptr,
                shape=(y_size,),
                strides=(1,),
                offsets=(y_offset,),
                block_shape=(1,),
                order=(0,),
            )
            mean = tl.load(running_mean_block_ptr)

        condition = tl.arange(0, x_block_size) < x_size
        centered_mean = tl.where(condition, input - mean, 0)

        if use_input_stats == 1:
            var_block_ptr = tl.make_block_ptr(
                var_ptr + batch * y_size,
                shape=(y_size,),
                strides=(1,),
                offsets=(y_offset,),
                block_shape=(1,),
                order=(0,),
            )
            var = (tl.sum(centered_mean * centered_mean / x_size, 1)).to(dtype)
            tl.store(var_block_ptr, var)
        else:
            running_var_block_ptr = tl.make_block_ptr(
                running_var_ptr,
                shape=(y_size,),
                strides=(1,),
                offsets=(y_offset,),
                block_shape=(1,),
                order=(0,),
            )
            var = tl.load(running_var_block_ptr)

        rstd = tl.math.rsqrt(var + eps)
        output = centered_mean * rstd

        if weight_ptr is not None:
            weight_block_ptr = tl.make_block_ptr(
                weight_ptr,
                shape=(y_size,),
                strides=(1,),
                offsets=(y_offset,),
                block_shape=(1,),
                order=(0,),
            )
            weight = tl.load(weight_block_ptr)
            output *= weight

        if bias_ptr is not None:
            bias_block_ptr = tl.make_block_ptr(
                bias_ptr,
                shape=(y_size,),
                strides=(1,),
                offsets=(y_offset,),
                block_shape=(1,),
                order=(0,),
            )
            bias = tl.load(bias_block_ptr)
            output += bias

        tl.store(output_block_ptr, output.to(dtype), boundary_check=(1,))

    @staticmethod
    @triton.jit
    def forward_running_mean_running_var(
        mean_ptr: tl.tensor,
        var_ptr: tl.tensor,
        running_mean_ptr: tl.tensor,
        running_var_ptr: tl.tensor,
        num_batches: tl.int32,
        y_size: tl.int32,
        momentum: tl.constexpr,
        batch_block_size: tl.constexpr,
    ):
        y_offset = tl.program_id(0)

        mean_block_ptr = tl.make_block_ptr(
            mean_ptr,
            shape=(num_batches, y_size),
            strides=(y_size, 1),
            offsets=(0, y_offset),
            block_shape=(batch_block_size, 1),
            order=(0, 1),
        )
        var_block_ptr = tl.make_block_ptr(
            var_ptr,
            shape=(num_batches, y_size),
            strides=(y_size, 1),
            offsets=(0, y_offset),
            block_shape=(batch_block_size, 1),
            order=(0, 1),
        )
        running_mean_block_ptr = tl.make_block_ptr(
            running_mean_ptr,
            shape=(y_size,),
            strides=(1,),
            offsets=(y_offset,),
            block_shape=(1,),
            order=(0,),
        )
        running_var_block_ptr = tl.make_block_ptr(
            running_var_ptr,
            shape=(y_size,),
            strides=(1,),
            offsets=(y_offset,),
            block_shape=(1,),
            order=(0,),
        )

        mean = tl.load(mean_block_ptr, boundary_check=(0,))
        mean = tl.sum(mean, 0) / num_batches
        var = tl.load(var_block_ptr, boundary_check=(0,))
        var = tl.sum(var, 0) / num_batches
        running_mean = tl.load(running_mean_block_ptr, boundary_check=(0,))
        running_mean = mean * momentum + running_mean * (1 - momentum)
        running_var = tl.load(running_var_block_ptr, boundary_check=(0,))
        running_var = var * momentum + running_var * (1 - momentum)
        tl.store(running_mean_block_ptr, running_mean, boundary_check=(0,))
        tl.store(running_var_block_ptr, running_var, boundary_check=(0,))

    @staticmethod
    @triton.jit
    def backward(
        grad_input_ptr: tl.tensor,
        grad_weight_staging_ptr: tl.tensor,
        grad_bias_staging_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        running_mean_ptr: tl.tensor,
        running_var_ptr: tl.tensor,
        mean_ptr: tl.tensor,
        var_ptr: tl.tensor,
        weight_ptr: tl.tensor,
        use_input_stats: tl.int32,
        eps: tl.float32,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch = pid // y_size
        y_offset = pid % y_size

        grad_input_block_ptr = tl.make_block_ptr(
            grad_input_ptr + batch * y_size * x_size,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr + batch * y_size * x_size,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr + batch * y_size * x_size,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(1,))
        input = tl.load(input_block_ptr, boundary_check=(1,))

        if use_input_stats == 1:
            mean_block_ptr = tl.make_block_ptr(
                mean_ptr + batch * y_size,
                shape=(y_size,),
                strides=(1,),
                offsets=(y_offset,),
                block_shape=(1,),
                order=(0,),
            )
            var_block_ptr = tl.make_block_ptr(
                var_ptr + batch * y_size,
                shape=(y_size,),
                strides=(1,),
                offsets=(y_offset,),
                block_shape=(1,),
                order=(0,),
            )
            mean = tl.load(mean_block_ptr)
            var = tl.load(var_block_ptr)
        else:
            running_mean_block_ptr = tl.make_block_ptr(
                running_mean_ptr,
                shape=(y_size,),
                strides=(1,),
                offsets=(y_offset,),
                block_shape=(1,),
                order=(0,),
            )
            running_var_block_ptr = tl.make_block_ptr(
                running_var_ptr,
                shape=(y_size,),
                strides=(1,),
                offsets=(y_offset,),
                block_shape=(1,),
                order=(0,),
            )
            mean = tl.load(running_mean_block_ptr)
            var = tl.load(running_var_block_ptr)

        condition = tl.arange(0, x_block_size) < x_size
        centered_mean = tl.where(condition, input - mean, 0)

        if weight_ptr is not None:
            weight_block_ptr = tl.make_block_ptr(
                weight_ptr,
                shape=(y_size,),
                strides=(1,),
                offsets=(y_offset,),
                block_shape=(1,),
                order=(0,),
            )
            weight = tl.load(weight_block_ptr)
            grad_norm = weight * grad_output
        else:
            grad_norm = grad_output

        rstd = tl.math.rsqrt(var + eps)
        grad_std = tl.sum(grad_norm * centered_mean, 1)
        grad_var = grad_std * -(0.5 * rstd * rstd * rstd) / x_size
        grad_distance = 2 * centered_mean * grad_var
        grad_centered_mean = tl.where(condition, grad_norm * rstd + grad_distance, 0)
        grad_mean = -tl.sum(grad_centered_mean, 1) / x_size
        grad_input = grad_centered_mean + grad_mean
        tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check=(1,))

        if grad_weight_staging_ptr is not None:
            norm = centered_mean * rstd
            grad_weight = norm * grad_output
            grad_weight_staging_block_ptr = tl.make_block_ptr(
                grad_weight_staging_ptr + batch * y_size,
                shape=(y_size,),
                strides=(1,),
                offsets=(y_offset,),
                block_shape=(1,),
                order=(0,),
            )
            grad_weight = tl.sum(grad_weight, 1)
            tl.store(grad_weight_staging_block_ptr, grad_weight.to(dtype))

        if grad_bias_staging_ptr is not None:
            grad_bias_staging_block_ptr = tl.make_block_ptr(
                grad_bias_staging_ptr + batch * y_size,
                shape=(y_size,),
                strides=(1,),
                offsets=(y_offset,),
                block_shape=(1,),
                order=(0,),
            )
            grad_bias = tl.sum(grad_output, 1)
            tl.store(grad_bias_staging_block_ptr, grad_bias.to(dtype))
