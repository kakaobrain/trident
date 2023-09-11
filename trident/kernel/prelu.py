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


def prelu_configs():
    configs = []
    for y_block_size in [32, 64, 128, 256]:
        for x_block_size in [1, 32, 64]:
            for num_warps in [4, 8, 16]:
                config = triton.Config({"y_block_size": y_block_size, "x_block_size": x_block_size}, num_warps)
                configs.append(config)
    return configs


class PReLU:
    @staticmethod
    @triton.autotune(prelu_configs(), ["x_size"])
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        weight_ptr: tl.tensor,
        num_batches: tl.int32,
        y_size: tl.int32,
        x_size: tl.int32,
        batch_stride: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        dtype: tl.constexpr,
        y_block_size: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_y_blocks = tl.cdiv(y_size, y_block_size)
        num_x_blocks = tl.cdiv(x_size, x_block_size)
        num_blocks = num_y_blocks * num_x_blocks
        batch_offset = pid // num_blocks
        block = pid % num_blocks
        y_block = block // num_x_blocks
        x_block = block % num_x_blocks
        y_offset = y_block * y_block_size
        x_offset = x_block * x_block_size

        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(num_batches, y_size, x_size),
            strides=(batch_stride, y_stride, x_stride),
            offsets=(batch_offset, y_offset, x_offset),
            block_shape=(1, y_block_size, x_block_size),
            order=(2, 1, 0),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(num_batches, y_size, x_size),
            strides=(batch_stride, y_stride, x_stride),
            offsets=(batch_offset, y_offset, x_offset),
            block_shape=(1, y_block_size, x_block_size),
            order=(2, 1, 0),
        )
        weight_block_ptr = tl.make_block_ptr(
            weight_ptr,
            shape=(y_size, 1),
            strides=(1, 0),
            offsets=(y_offset, 0),
            block_shape=(y_block_size, 1),
            order=(1, 0),
        )

        input = tl.load(input_block_ptr, boundary_check=(1, 2))
        weight = tl.load(weight_block_ptr, boundary_check=(0,))
        output = language.math.LeakyReLU.forward(input, weight)
        tl.store(output_block_ptr, output.to(dtype), boundary_check=(1, 2))

    @staticmethod
    @triton.autotune(prelu_configs(), ["x_size"])
    @triton.jit
    def backward(
        grad_input_ptr: tl.tensor,
        grad_weight_staging_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        weight_ptr: tl.tensor,
        num_batches: tl.int32,
        y_size: tl.int32,
        x_size: tl.int32,
        batch_stride: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        y_block_size: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_y_blocks = tl.cdiv(y_size, y_block_size)
        num_x_blocks = tl.cdiv(x_size, x_block_size)
        num_blocks = num_y_blocks * num_x_blocks
        batch_offset = pid // num_blocks
        block = pid % num_blocks
        y_block = block // num_x_blocks
        x_block = block % num_x_blocks
        y_offset = y_block * y_block_size
        x_offset = x_block * x_block_size

        grad_input_block_ptr = tl.make_block_ptr(
            grad_input_ptr,
            shape=(num_batches, y_size, x_size),
            strides=(batch_stride, y_stride, x_stride),
            offsets=(batch_offset, y_offset, x_offset),
            block_shape=(1, y_block_size, x_block_size),
            order=(2, 1, 0),
        )
        grad_weight_staging_block_ptr = tl.make_block_ptr(
            grad_weight_staging_ptr,
            shape=(num_batches, y_size, x_size),
            strides=(batch_stride, y_stride, x_stride),
            offsets=(batch_offset, y_offset, x_offset),
            block_shape=(1, y_block_size, x_block_size),
            order=(2, 1, 0),
        )
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(num_batches, y_size, x_size),
            strides=(batch_stride, y_stride, x_stride),
            offsets=(batch_offset, y_offset, x_offset),
            block_shape=(1, y_block_size, x_block_size),
            order=(2, 1, 0),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(num_batches, y_size, x_size),
            strides=(batch_stride, y_stride, x_stride),
            offsets=(batch_offset, y_offset, x_offset),
            block_shape=(1, y_block_size, x_block_size),
            order=(2, 1, 0),
        )
        weight_block_ptr = tl.make_block_ptr(
            weight_ptr,
            shape=(y_size, 1),
            strides=(1, 0),
            offsets=(y_offset, 0),
            block_shape=(y_block_size, 1),
            order=(1, 0),
        )

        input = tl.load(input_block_ptr, boundary_check=(1, 2))
        weight = tl.load(weight_block_ptr)
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(1, 2))
        grad_input = language.math.LeakyReLU.backward(grad_output, input, weight)
        grad_weight = grad_output * tl.where(input > 0, 0, input)
        tl.store(grad_input_block_ptr, grad_input, boundary_check=(1, 2))
        tl.store(grad_weight_staging_block_ptr, grad_weight, boundary_check=(1, 2))
