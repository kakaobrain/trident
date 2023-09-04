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
        output_ptr: tl.tensor,
        argmax_ptr: tl.tensor,
        input_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        x_block_size: tl.constexpr,
    ):
        y_offset = tl.program_id(0)
        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(y_size,),
            strides=(1,),
            offsets=(y_offset,),
            block_shape=(1,),
            order=(0,),
        )
        argmax_block_ptr = tl.make_block_ptr(
            argmax_ptr,
            shape=(y_size,),
            strides=(1,),
            offsets=(y_offset,),
            block_shape=(1,),
            order=(0,),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        input = tl.load(input_block_ptr, boundary_check=(1,))
        argmax = tl.argmax(input, 1)
        output = tl.load(input_ptr + y_offset * y_stride + argmax * x_stride)
        tl.store(output_block_ptr, output)
        tl.store(argmax_block_ptr, argmax.to(tl.int64))

    @staticmethod
    @triton.jit
    def backward(
        grad_input_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        argmax_ptr: tl.tensor,
        y_size: tl.constexpr,
        x_size: tl.constexpr,
        y_stride: tl.constexpr,
        x_stride: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        y_offset = tl.program_id(0)
        grad_input_block_ptr = tl.make_block_ptr(
            grad_input_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(y_size,),
            strides=(1,),
            offsets=(y_offset,),
            block_shape=(1,),
            order=(0,),
        )
        argmax_block_ptr = tl.make_block_ptr(
            argmax_ptr,
            shape=(1, y_size),
            strides=(y_size, 1),
            offsets=(0, y_offset),
            block_shape=(1, 1),
            order=(1, 0),
        )
        grad_output = tl.load(grad_output_block_ptr)
        argmax = tl.load(argmax_block_ptr)
        condition = tl.arange(0, x_block_size) == argmax
        grad_input = tl.where(condition, grad_output, 0)
        tl.store(grad_input_block_ptr, grad_input, boundary_check=(1,))
