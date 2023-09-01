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

from trident import kernel, language


class LayerNorm:
    @staticmethod
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
        rstd_ptr: tl.tensor,
        mean_ptr: tl.tensor,
        input_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        weight_ptr: tl.tensor,
        bias_ptr: tl.tensor,
        eps: tl.float32,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        y_offset = tl.program_id(0)
        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        rstd_block_ptr = tl.make_block_ptr(
            rstd_ptr,
            shape=(y_size,),
            strides=(1,),
            offsets=(y_offset,),
            block_shape=(1,),
            order=(0,),
        )
        mean_block_ptr = tl.make_block_ptr(
            mean_ptr,
            shape=(y_size,),
            strides=(1,),
            offsets=(y_offset,),
            block_shape=(1,),
            order=(0,),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )

        input = tl.load(input_block_ptr, boundary_check=(1,), padding_option="zero")
        mean = tl.sum(input, 1) / x_size
        condition = tl.arange(0, x_block_size) < x_size
        centered_mean = tl.where(condition, input - mean, 0)
        var = tl.sum(centered_mean * centered_mean, 1) / x_size
        rstd = tl.math.rsqrt(var + eps)
        output = centered_mean * rstd

        if weight_ptr is not None:
            weight_block_ptr = tl.make_block_ptr(
                weight_ptr,
                shape=(x_size,),
                strides=(1,),
                offsets=(0,),
                block_shape=(x_block_size,),
                order=(0,),
            )
            weight = tl.load(weight_block_ptr, boundary_check=(0,))
            output *= weight

        if bias_ptr is not None:
            bias_block_ptr = tl.make_block_ptr(
                bias_ptr,
                shape=(x_size,),
                strides=(1,),
                offsets=(0,),
                block_shape=(x_block_size,),
                order=(0,),
            )
            bias = tl.load(bias_block_ptr, boundary_check=(0,))
            output += bias

        tl.store(output_block_ptr, output.to(dtype), boundary_check=(1,))
        tl.store(rstd_block_ptr, rstd.to(dtype))
        tl.store(mean_block_ptr, mean.to(dtype))

    @staticmethod
    @triton.jit
    def backward(
        grad_input_ptr: tl.tensor,
        grad_weight_staging_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        weight_ptr: tl.tensor,
        rstd_ptr: tl.tensor,
        mean_ptr: tl.tensor,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        y_offset = tl.program_id(0)
        grad_input_block_ptr = tl.make_block_ptr(
            grad_input_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        rstd_block_ptr = tl.make_block_ptr(
            rstd_ptr,
            shape=(y_size,),
            strides=(1,),
            offsets=(y_offset,),
            block_shape=(1,),
            order=(0,),
        )
        mean_block_ptr = tl.make_block_ptr(
            mean_ptr,
            shape=(y_size,),
            strides=(1,),
            offsets=(y_offset,),
            block_shape=(1,),
            order=(0,),
        )
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(1,))
        input = tl.load(input_block_ptr, boundary_check=(1,))
        rstd = tl.load(rstd_block_ptr)
        mean = tl.load(mean_block_ptr)
        condition = tl.arange(0, x_block_size) < x_size
        centered_mean = tl.where(condition, input - mean, 0)

        if weight_ptr is not None:
            weight_block_ptr = tl.make_block_ptr(
                weight_ptr,
                shape=(1, x_size),
                strides=(x_size, 1),
                offsets=(0, 0),
                block_shape=(1, x_block_size),
                order=(1, 0),
            )
            weight = tl.load(weight_block_ptr, boundary_check=(1,))
            grad_norm = weight * grad_output
        else:
            grad_norm = grad_output

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
                grad_weight_staging_ptr,
                shape=(y_size, x_size),
                strides=(x_size, 1),
                offsets=(y_offset, 0),
                block_shape=(1, x_block_size),
                order=(1, 0),
            )
            tl.store(grad_weight_staging_block_ptr, grad_weight.to(dtype), boundary_check=(1,))
