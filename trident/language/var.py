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


class Var:
    @staticmethod
    @triton.jit
    def forward(
        input_ptr: tl.tensor,
        y_size: int,
        x_size: int,
        y_stride: int,
        x_stride: int,
        y_offset: int,
        mean: tl.tensor,
        correction: tl.constexpr,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        output = tl.zeros((1, x_block_size), tl.float32)

        for block_offset in range(0, x_size, x_block_size):
            input = tl.load(input_block_ptr, boundary_check=(1,))
            mask = (tl.arange(0, x_block_size) + block_offset) < x_size
            input = tl.where(mask, input - mean, 0.0)
            output += tl.math.pow(input, 2.0)
            input_block_ptr = tl.advance(input_block_ptr, (0, x_block_size))

        output = tl.sum(output, 1) / (x_size - correction)

        return output.to(dtype)

    @staticmethod
    @triton.jit
    def backward(
        input_ptr: tl.tensor,
        y_size: int,
        x_size: int,
        y_stride: int,
        x_stride: int,
        y_offset: int,
        x_offset: int,
        mean: tl.tensor,
        correction: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        input = tl.load(input_block_ptr, boundary_check=(1,), padding_option="zero")
        condition = tl.arange(0, x_block_size) + x_offset < x_size
        centered_mean = tl.where(condition[None, :], input - mean, 0.0) / (x_size - correction)
        grad_input = 2 * centered_mean / x_size

        return grad_input
