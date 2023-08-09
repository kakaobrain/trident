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

from trident import kernel, language


class LayerNorm:
    @staticmethod
    @triton.jit
    def forward(
        output_ptr,
        input_ptr,
        y_size,
        x_size,
        weight_ptr,
        bias_ptr,
        eps,
        block_size: triton.language.constexpr,
        dtype: triton.language.constexpr,
    ):
        offset = triton.language.program_id(0)

        mean = language.mean(
            input_ptr,
            y_size,
            x_size,
            offset,
            language.dim[1],
            block_size,
            dtype,
        )
        var = language.var(
            input_ptr,
            y_size,
            x_size,
            offset,
            mean,
            language.dim[1],
            language.zero,
            block_size,
            dtype,
        )
        std = language.std(var, eps)

        input_block_ptr = triton.language.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(offset, 0),
            block_shape=(1, block_size),
            order=(1, 0),
        )
        output_block_ptr = triton.language.make_block_ptr(
            output_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(offset, 0),
            block_shape=(1, block_size),
            order=(1, 0),
        )

        for block_offset in range(0, x_size, block_size):
            input = triton.language.load(input_block_ptr, boundary_check=(1,))
            output = language.norm(input, mean, std)

            if weight_ptr is not None:
                weight_block_ptr = triton.language.make_block_ptr(
                    weight_ptr,
                    shape=(1, x_size),
                    strides=(x_size, 1),
                    offsets=(0, block_offset),
                    block_shape=(1, block_size),
                    order=(1, 0),
                )
                weight = triton.language.load(weight_block_ptr, boundary_check=(1,))
                output *= weight

            if bias_ptr is not None:
                bias_block_ptr = triton.language.make_block_ptr(
                    bias_ptr,
                    shape=(1, x_size),
                    strides=(x_size, 1),
                    offsets=(0, block_offset),
                    block_shape=(1, block_size),
                    order=(1, 0),
                )
                bias = triton.language.load(bias_block_ptr, boundary_check=(1,))
                output += bias

            triton.language.store(
                output_block_ptr, output.to(dtype), boundary_check=(1,)
            )
            input_block_ptr = triton.language.advance(input_block_ptr, (0, block_size))
            output_block_ptr = triton.language.advance(
                output_block_ptr, (0, block_size)
            )

    @staticmethod
    @triton.jit
    def backward(
        grad_input_ptr,
        grad_weight_staging_ptr,
        grad_output_ptr,
        input_ptr,
        y_size,
        x_size,
        weight_ptr,
        eps,
        block_size: triton.language.constexpr,
        dtype: triton.language.constexpr,
    ):
        offset = triton.language.program_id(0)

        mean = language.mean(
            input_ptr,
            y_size,
            x_size,
            offset,
            language.dim[1],
            block_size,
            dtype,
        )
        var = language.var(
            input_ptr,
            y_size,
            x_size,
            offset,
            mean,
            language.dim[1],
            language.zero,
            block_size,
            dtype,
        )
        std = language.std(var, eps)

        input_block_ptr = triton.language.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(offset, 0),
            block_shape=(1, block_size),
            order=(1, 0),
        )
        input = triton.language.load(input_block_ptr, boundary_check=(1,))
        condition = triton.language.arange(0, block_size) < x_size
        centered_mean = triton.language.where(condition, input - mean, 0)

        grad_output_block_ptr = triton.language.make_block_ptr(
            grad_output_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(0, 0),
            block_shape=(1, block_size),
            order=(1, 0),
        )
        grad_output = triton.language.load(grad_output_block_ptr, boundary_check=(1,))

        if weight_ptr is not None:
            weight_block_ptr = triton.language.make_block_ptr(
                weight_ptr,
                shape=(1, x_size),
                strides=(x_size, 1),
                offsets=(0, 0),
                block_shape=(1, block_size),
                order=(1, 0),
            )
            weight = triton.language.load(weight_block_ptr, boundary_check=(1,))
            grad_norm = weight * grad_output
        else:
            grad_norm = grad_output

        grad_std = ((grad_norm * centered_mean) / -language.pow2(std)) / (2 * std)
        grad_var = triton.language.sum(grad_std, 1) / x_size
        grad_distance = 2 * centered_mean * grad_var
        grad_centered_mean = triton.language.where(
            condition, (grad_norm / std) + grad_distance, 0
        )
        grad_mean = -triton.language.sum(grad_centered_mean, 1) / x_size
        grad_input = grad_centered_mean + grad_mean

        grad_input_block_ptr = triton.language.make_block_ptr(
            grad_input_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(offset, 0),
            block_shape=(1, block_size),
            order=(1, 0),
        )
        triton.language.store(
            grad_input_block_ptr, grad_input.to(dtype), boundary_check=(1,)
        )

        if grad_weight_staging_ptr is not None:
            norm = centered_mean / std
            grad_weight = norm * grad_output
            grad_weight_staging_block_ptr = triton.language.make_block_ptr(
                grad_weight_staging_ptr,
                shape=(y_size, x_size),
                strides=(x_size, 1),
                offsets=(offset, 0),
                block_shape=(1, block_size),
                order=(1, 0),
            )
            triton.language.store(
                grad_weight_staging_block_ptr,
                grad_weight.to(dtype),
                boundary_check=(1,),
            )