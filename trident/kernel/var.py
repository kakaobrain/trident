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

from trident import language


class Var:
    @staticmethod
    @triton.jit
    def forward(
        output_ptr,
        input_ptr,
        y_size,
        x_size,
        average,
        dim: triton.language.constexpr,
        correction: triton.language.constexpr,
        block_size: triton.language.constexpr,
        dtype: triton.language.constexpr,
    ):
        offset = triton.language.program_id(0)
        output = language.var(
            input_ptr,
            y_size,
            x_size,
            offset,
            average,
            dim,
            correction,
            block_size,
            dtype,
        )
        output_block_ptr = triton.language.make_block_ptr(
            output_ptr,
            shape=(y_size if dim == 0 else x_size,),
            strides=(1,),
            offsets=(offset,),
            block_shape=(1,),
            order=(0,),
        )
        triton.language.store(output_block_ptr, output)

    @staticmethod
    @triton.jit
    def backward(
        grad_input_ptr,
        grad_output_ptr,
        input_ptr,
        y_size,
        x_size,
        dim: triton.language.constexpr,
        correction: triton.language.constexpr,
        block_size: triton.language.constexpr,
        dtype: triton.language.constexpr,
    ):
        offset = triton.language.program_id(0)

        if dim == 0:
            grad_input_block_ptr = triton.language.make_block_ptr(
                grad_input_ptr,
                shape=(y_size, x_size),
                strides=(x_size, 1),
                offsets=(0, offset),
                block_shape=(block_size, 1),
                order=(0, 1),
            )
            input_block_ptr = triton.language.make_block_ptr(
                input_ptr,
                shape=(y_size, x_size),
                strides=(x_size, 1),
                offsets=(0, offset),
                block_shape=(block_size, 1),
                order=(0, 1),
            )
            grad_output_size = x_size
            size_along_dim = y_size
        else:
            grad_input_block_ptr = triton.language.make_block_ptr(
                grad_input_ptr,
                shape=(y_size, x_size),
                strides=(x_size, 1),
                offsets=(offset, 0),
                block_shape=(1, block_size),
                order=(1, 0),
            )
            input_block_ptr = triton.language.make_block_ptr(
                input_ptr,
                shape=(y_size, x_size),
                strides=(x_size, 1),
                offsets=(offset, 0),
                block_shape=(1, block_size),
                order=(1, 0),
            )
            grad_output_size = y_size
            size_along_dim = x_size

        grad_output = language.sum(
            grad_output_ptr,
            1,
            grad_output_size,
            0,
            language.dim[1],
            block_size,
            dtype,
        ) / (grad_output_size - correction)
        average = language.mean(
            input_ptr, y_size, x_size, offset, dim, block_size, dtype
        )

        for block_offset in range(0, size_along_dim, block_size):
            input = triton.language.load(
                input_block_ptr, boundary_check=(dim,), padding_option="zero"
            )
            mask = (
                triton.language.arange(0, block_size) + block_offset
            ) < size_along_dim
            centered_mean = triton.language.where(
                mask[:, None] if dim == 0 else mask[None, :], input - average, 0.0
            )
            grad_input = grad_output * 2 * centered_mean / size_along_dim
            triton.language.store(
                grad_input_block_ptr, grad_input.to(dtype), boundary_check=(dim,)
            )
            input_block_ptr = triton.language.advance(
                input_block_ptr, (block_size, 0) if dim == 0 else (0, block_size)
            )
