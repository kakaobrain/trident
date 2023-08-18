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


class Softmax:
    def configs():
        configs = []
        for num_warps in [4, 8, 16]:
            for block_size in [128, 256, 512, 1024, 2048]:
                config = triton.Config(
                    {"block_size": block_size},
                    num_warps=num_warps,
                )
                configs.append(config)
        return configs

    @staticmethod
    @triton.autotune(
        configs=configs(),
        key=["y_size", "x_size", "dim"],
    )
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        y_size: int,
        x_size: int,
        dim: tl.constexpr,
        block_size: tl.constexpr,
        dtype: tl.constexpr,
    ):
        offset = tl.program_id(0)

        if dim == 0:
            input_block_ptr = tl.make_block_ptr(
                input_ptr,
                shape=(x_size, y_size),
                strides=(1, x_size),
                offsets=(offset, 0),
                block_shape=(1, block_size),
                order=(0, 1),
            )
            size_along_dim = y_size
        else:
            input_block_ptr = tl.make_block_ptr(
                input_ptr,
                shape=(y_size, x_size),
                strides=(x_size, 1),
                offsets=(offset, 0),
                block_shape=(1, block_size),
                order=(1, 0),
            )
            size_along_dim = x_size

        max = tl.full((1, block_size), -float("inf"), tl.float32)
        sum = tl.zeros((1, block_size), tl.float32)

        for block_offset in range(0, size_along_dim, block_size):
            input = tl.load(input_block_ptr, boundary_check=(1,)).to(tl.float32)
            condition = tl.arange(0, block_size) + block_offset < size_along_dim
            input = tl.where(condition, input, -float("inf"))
            peak = tl.maximum(max, input)
            peak = tl.where(condition, peak, 0)
            sum = sum * tl.exp(max - peak) + tl.exp(input - peak)
            max = peak
            input_block_ptr = tl.advance(input_block_ptr, (0, block_size))

        max, sum = tl.reduce((max, sum), 1, language.combine_softmax)

        if dim == 0:
            input_block_ptr = tl.make_block_ptr(
                input_ptr,
                shape=(x_size, y_size),
                strides=(1, x_size),
                offsets=(offset, 0),
                block_shape=(1, block_size),
                order=(0, 1),
            )
            output_block_ptr = tl.make_block_ptr(
                output_ptr,
                shape=(x_size, y_size),
                strides=(1, x_size),
                offsets=(offset, 0),
                block_shape=(1, block_size),
                order=(0, 1),
            )
        else:
            input_block_ptr = tl.make_block_ptr(
                input_ptr,
                shape=(y_size, x_size),
                strides=(x_size, 1),
                offsets=(offset, 0),
                block_shape=(1, block_size),
                order=(1, 0),
            )
            output_block_ptr = tl.make_block_ptr(
                output_ptr,
                shape=(y_size, x_size),
                strides=(x_size, 1),
                offsets=(offset, 0),
                block_shape=(1, block_size),
                order=(1, 0),
            )

        for _ in range(0, size_along_dim, block_size):
            input = tl.load(input_block_ptr, boundary_check=(1,)).to(tl.float32)
            output = tl.exp(input - max) / sum
            tl.store(output_block_ptr, output.to(dtype), boundary_check=(1,))
            input_block_ptr = tl.advance(input_block_ptr, (0, block_size))
            output_block_ptr = tl.advance(output_block_ptr, (0, block_size))

    @staticmethod
    @triton.autotune(
        configs=configs(),
        key=["y_size", "x_size"],
    )
    @triton.jit
    def backward_delta(
        delta_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        output_ptr: tl.tensor,
        x_size: int,
        y_size: int,
        x_stride: int,
        y_stride: int,
        block_size: tl.constexpr,
    ):
        offset = tl.program_id(0)

        delta_block_ptr = tl.make_block_ptr(
            delta_ptr,
            shape=(y_size,),
            strides=(1,),
            offsets=(offset,),
            block_shape=(1,),
            order=(0,),
        )
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(offset, 0),
            block_shape=(1, block_size),
            order=(1, 0),
        )
        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(offset, 0),
            block_shape=(1, block_size),
            order=(1, 0),
        )

        delta = tl.zeros((1, block_size), tl.float32)

        for _ in range(0, x_size, block_size):
            output = tl.load(output_block_ptr, boundary_check=(1,), padding_option="zero").to(tl.float32)
            grad_output = tl.load(grad_output_block_ptr, boundary_check=(1,), padding_option="zero").to(tl.float32)
            delta += output * grad_output
            output_block_ptr = tl.advance(output_block_ptr, (0, block_size))
            grad_output_block_ptr = tl.advance(grad_output_block_ptr, (0, block_size))

        tl.store(delta_block_ptr, tl.sum(delta, 1))

    @staticmethod
    @triton.autotune(
        configs=configs(),
        key=["y_size", "x_size"],
    )
    @triton.jit
    def backward(
        grad_input_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        output_ptr: tl.tensor,
        delta_ptr: tl.tensor,
        x_size: int,
        y_size: int,
        x_stride: int,
        y_stride: int,
        block_size: tl.constexpr,
        dtype: tl.constexpr,
    ):
        offset = tl.program_id(0)

        grad_input_block_ptr = tl.make_block_ptr(
            grad_input_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(offset, 0),
            block_shape=(1, block_size),
            order=(1, 0),
        )
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(offset, 0),
            block_shape=(1, block_size),
            order=(1, 0),
        )
        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(offset, 0),
            block_shape=(1, block_size),
            order=(1, 0),
        )
        delta_block_ptr = tl.make_block_ptr(
            delta_ptr,
            shape=(y_size,),
            strides=(1,),
            offsets=(offset,),
            block_shape=(1,),
            order=(0,),
        )

        grad_input = tl.zeros((1, block_size), tl.float32)
        for block_offset in range(0, x_size, block_size):
            output = tl.load(output_block_ptr, boundary_check=(1,), padding_option="zero").to(tl.float32)
            grad_output = tl.load(grad_output_block_ptr, boundary_check=(1,), padding_option="zero").to(tl.float32)
            delta = tl.load(delta_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
            grad_input = output * (grad_output - delta)
            tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check=(1,))
            output_block_ptr = tl.advance(output_block_ptr, (0, block_size))
            grad_output_block_ptr = tl.advance(grad_output_block_ptr, (0, block_size))
            grad_input_block_ptr = tl.advance(grad_input_block_ptr, (0, block_size))
