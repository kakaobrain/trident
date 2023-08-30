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


class Var:
    @staticmethod
    def configs():
        configs = []
        for x_block_size in [512, 1024, 2048, 4096]:
            for num_stages in [4, 5]:
                config = triton.Config({"x_block_size": x_block_size}, 2 if x_block_size <= 512 else 4, num_stages)
                configs.append(config)
        return configs

    @staticmethod
    @triton.autotune(configs(), ["x_size"])
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        y_size: int,
        x_size: int,
        y_stride: int,
        x_stride: int,
        correction: tl.constexpr,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        y_offset = tl.program_id(0)
        output, mean = language.VarMean.forward(
            input_ptr,
            y_size,
            x_size,
            y_stride,
            x_stride,
            y_offset,
            correction,
            dtype,
            x_block_size,
        )
        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(y_size,),
            strides=(1,),
            offsets=(y_offset,),
            block_shape=(1,),
            order=(0,),
        )
        tl.store(output_block_ptr, output)

    @staticmethod
    @triton.autotune(configs(), ["x_size"])
    @triton.jit
    def backward(
        grad_input_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        y_size: int,
        x_size: int,
        y_stride: int,
        x_stride: int,
        correction: tl.constexpr,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_x_blocks = tl.cdiv(x_size, x_block_size)
        y_offset = pid // num_x_blocks
        xid = pid % num_x_blocks
        x_offset = xid * x_block_size

        grad_input_block_ptr = tl.make_block_ptr(
            grad_input_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, x_offset),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        mean = language.Mean.forward(input_ptr, y_size, x_size, y_stride, x_stride, y_offset, dtype, x_block_size)
        grad_input = language.Var.backward(
            grad_output_ptr,
            input_ptr,
            y_size,
            x_size,
            y_stride,
            x_stride,
            y_offset,
            x_offset,
            mean,
            correction,
            dtype,
            x_block_size,
        )
        tl.store(grad_input_block_ptr, grad_input, boundary_check=(1,))
