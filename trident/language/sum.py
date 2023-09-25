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


class Sum:
    @staticmethod
    @triton.jit
    def forward(
        input_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        y_offset: tl.int32,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
        require_x_boundary_check: tl.constexpr,
    ):
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )

        sum = tl.zeros((1, x_block_size), dtype)

        for _ in range(0, x_size, x_block_size):
            if require_x_boundary_check:
                input = tl.load(input_block_ptr, boundary_check=(1,), padding_option="zero")
            else:
                input = tl.load(input_block_ptr)

            sum += input
            input_block_ptr = tl.advance(input_block_ptr, (0, x_block_size))

        return tl.sum(sum, 1).to(dtype)

    @staticmethod
    @triton.jit
    def backward(
        grad_output_ptr: tl.tensor,
        y_size: tl.int32,
        y_offset: tl.int32,
        x_block_size: tl.constexpr,
    ):
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(y_size, 1),
            strides=(1, 0),
            offsets=(y_offset, 0),
            block_shape=(1, 1),
            order=(0, 1),
        )

        grad_output = tl.load(grad_output_block_ptr)
        grad_input = tl.broadcast_to(grad_output, (1, x_block_size))

        return grad_input
