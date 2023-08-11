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

from trident import language


class Max:
    @staticmethod
    @triton.jit
    def forward(
        output_ptr,
        indices_ptr,
        input_ptr,
        y_size,
        x_size,
        dim: tl.constexpr,
        block_size: tl.constexpr,
        dtype: tl.constexpr,
    ):
        offset = tl.program_id(0)
        output, index = language.max(
            input_ptr,
            y_size,
            x_size,
            offset,
            dim,
            block_size,
            dtype,
        )
        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(y_size if dim == 0 else x_size,),
            strides=(1,),
            offsets=(offset,),
            block_shape=(1,),
            order=(0,),
        )
        indices_block_ptr = tl.make_block_ptr(
            indices_ptr,
            shape=(y_size if dim == 0 else x_size,),
            strides=(1,),
            offsets=(offset,),
            block_shape=(1,),
            order=(0,),
        )
        tl.store(output_block_ptr, output)
        tl.store(indices_block_ptr, index)

    @staticmethod
    @triton.jit
    def backward(
        grad_input_ptr,
        grad_output_ptr,
        indices_ptr,
        y_size,
        x_size,
        dim: tl.constexpr,
        block_size: tl.constexpr,
        dtype: tl.constexpr,
    ):
        offset = tl.program_id(0)

        if dim == 0:
            grad_input_block_ptr = tl.make_block_ptr(
                grad_input_ptr,
                shape=(x_size, y_size),
                strides=(1, x_size),
                offsets=(offset, 0),
                block_shape=(1, block_size),
                order=(1, 0),
            )
            size_along_dim = y_size
            grad_output_size = x_size
        else:
            grad_input_block_ptr = tl.make_block_ptr(
                grad_input_ptr,
                shape=(y_size, x_size),
                strides=(x_size, 1),
                offsets=(offset, 0),
                block_shape=(1, block_size),
                order=(1, 0),
            )
            size_along_dim = x_size
            grad_output_size = y_size

        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(1, grad_output_size),
            strides=(grad_output_size, 1),
            offsets=(0, offset),
            block_shape=(1, 1),
            order=(1, 0),
        )
        indices_block_ptr = tl.make_block_ptr(
            indices_ptr,
            shape=(1, grad_output_size),
            strides=(grad_output_size, 1),
            offsets=(0, offset),
            block_shape=(1, 1),
            order=(1, 0),
        )
        index = tl.load(indices_block_ptr)
        grad_output = tl.load(grad_output_block_ptr)

        for block_offset in range(0, size_along_dim, block_size):
            condition = tl.arange(0, block_size) + block_offset == index
            condition = condition.to(tl.float32)
            grad_input = tl.where(condition, grad_output, 0)
            tl.store(grad_input_block_ptr, grad_input, boundary_check=(1,))
            grad_input_block_ptr = tl.advance(grad_input_block_ptr, (0, block_size))
