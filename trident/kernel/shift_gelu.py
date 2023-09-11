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

from trident import language, util


def shift_gelu_configs():
    configs = []
    for x_block_size in [256, 512, 1024, 2048]:
        for num_stages in [4, 5]:
            config = triton.Config({"x_block_size": x_block_size}, 2 if x_block_size <= 512 else 4, num_stages)
            configs.append(config)
    return configs


class ShiftGELU:
    @staticmethod
    @util.autotune(shift_gelu_configs(), ["x_size"])
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
        shift_ptr: tl.tensor,
        input_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        bias_ptr: tl.tensor,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_x_blocks = tl.cdiv(x_size, x_block_size)
        y_offset = pid // num_x_blocks
        x = pid % num_x_blocks
        x_offset = x * x_block_size

        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(y_offset, x_offset),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        shift_block_ptr = tl.make_block_ptr(
            shift_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(y_offset, x_offset),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(y_offset, x_offset),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        bias_block_ptr = tl.make_block_ptr(
            bias_ptr,
            shape=(1, x_size),
            strides=(x_size, 1),
            offsets=(0, x_offset),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )

        input = tl.load(input_block_ptr, boundary_check=(1,))
        bias = tl.load(bias_block_ptr, boundary_check=(1,))
        shift = input + bias
        output = language.math.GELU.forward(shift)
        tl.store(output_block_ptr, output.to(dtype), boundary_check=(1,))
        tl.store(shift_block_ptr, shift.to(dtype), boundary_check=(1,))

    @staticmethod
    @util.autotune(shift_gelu_configs(), ["x_size"])
    @triton.jit
    def backward(
        grad_input_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        shift_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_x_blocks = tl.cdiv(x_size, x_block_size)
        y_offset = pid // num_x_blocks
        x = pid % num_x_blocks
        x_offset = x * x_block_size

        grad_input_block_ptr = tl.make_block_ptr(
            grad_input_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(y_offset, x_offset),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(y_offset, x_offset),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        shift_block_ptr = tl.make_block_ptr(
            shift_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(y_offset, x_offset),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )

        grad_output = tl.load(grad_output_block_ptr, boundary_check=(1,))
        shift = tl.load(shift_block_ptr, boundary_check=(1,))
        grad_input = language.math.GELU.backward(grad_output, shift)
        tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check=(1,))
