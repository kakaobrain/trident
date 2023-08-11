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


class Sum:
    @staticmethod
    @triton.jit
    def forward(
        output_ptr,
        input_ptr,
        y_size,
        x_size,
        dim: tl.constexpr,
        block_size: tl.constexpr,
        dtype: tl.constexpr,
    ):
        offset = tl.program_id(0)
        output = language.sum(input_ptr, y_size, x_size, offset, dim, block_size, dtype)
        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(y_size if dim == 0 else x_size,),
            strides=(1,),
            offsets=(offset,),
            block_shape=(1,),
            order=(0,),
        )
        tl.store(output_block_ptr, output)

    @staticmethod
    @triton.jit
    def backward(
        grad_input_ptr,
        grad_output_ptr,
        y_size,
        x_size,
        dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offset = tl.program_id(0)

        if dim == 0:
            grad_input_block_ptr = tl.make_block_ptr(
                grad_input_ptr,
                shape=(y_size, x_size),
                strides=(x_size, 1),
                offsets=(offset, 0),
                block_shape=(1, block_size),
                order=(1, 0),
            )
            dim = x_size
        else:
            grad_input_block_ptr = tl.make_block_ptr(
                grad_input_ptr,
                shape=(x_size, y_size),
                strides=(1, x_size),
                offsets=(offset, 0),
                block_shape=(1, block_size),
                order=(0, 1),
            )
            dim = y_size

        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(1, dim),
            strides=(dim, 1),
            offsets=(0, 0),
            block_shape=(1, block_size),
            order=(1, 0),
        )

        for _ in range(0, dim, block_size):
            grad_output = tl.load(grad_output_block_ptr)
            tl.store(grad_input_block_ptr, grad_output, boundary_check=(1,))
            grad_input_block_ptr = tl.advance(grad_input_block_ptr, (0, block_size))
            grad_output_block_ptr = tl.advance(grad_output_block_ptr, (0, block_size))
