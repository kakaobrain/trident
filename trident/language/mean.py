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


class Mean:
    @staticmethod
    @triton.jit
    def forward(
        input_ptr: tl.tensor,
        y_size: int,
        x_size: int,
        y_stride: int,
        x_stride: int,
        y_offset: int,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        sum = language.Sum.forward(input_ptr, y_size, x_size, y_stride, x_stride, y_offset, dtype, x_block_size, True)
        mean = sum / x_size

        return mean.to(dtype)

    @staticmethod
    @triton.jit
    def backward(
        grad_output_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        y_offset: tl.int32,
        dtype: tl.constexpr,
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
        return tl.broadcast_to(grad_output * 1.0 / x_size, (1, x_block_size)).to(dtype)
