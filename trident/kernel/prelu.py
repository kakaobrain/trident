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


def all_configs():
    configs = []

    for y_block_size in [32, 64, 128]:
        for x_block_size in [32, 64, 128]:
            for num_stages in [4, 5]:
                for num_warps in [4, 8, 16]:
                    configs.append(
                        triton.Config(
                            {
                                "y_block_size": y_block_size,
                                "x_block_size": x_block_size,
                            },
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )

    return configs


class PReLU:
    @staticmethod
    @triton.autotune(
        configs=all_configs(),
        key=["y_size", "x_size"],
    )
    @triton.jit
    def forward(
        output_ptr,
        input_ptr,
        weight_ptr,
        y_size,
        x_size,
        y_block_size: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_x_blocks = tl.cdiv(x_size, x_block_size)
        i = pid // num_x_blocks
        j = pid % num_x_blocks

        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(i * y_block_size, j * x_block_size),
            block_shape=(y_block_size, x_block_size),
            order=(1, 0),
        )

        weight_block_ptr = tl.make_block_ptr(
            weight_ptr,
            shape=x_size,
            strides=1,
            offsets=j * x_block_size,
            block_shape=x_block_size,
            order=(0,),
        )

        input = tl.load(input_block_ptr, boundary_check=(0, 1))
        weight = tl.load(weight_block_ptr, boundary_check=(0,))
        output = language.leaky_relu(input, weight)

        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(i * y_block_size, j * x_block_size),
            block_shape=(y_block_size, x_block_size),
            order=(1, 0),
        )

        tl.store(output_block_ptr, output.to(input.dtype), boundary_check=(0, 1))

    @staticmethod
    @triton.autotune(
        configs=all_configs(),
        key=["y_size", "x_size"],
    )
    @triton.jit
    def backward(
        grad_input_ptr,
        grad_weight_ptr,
        input_ptr,
        weight_ptr,
        grad_output_ptr,
        y_size,
        x_size,
        y_block_size: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_x_blocks = tl.cdiv(x_size, x_block_size)
        i = pid // num_x_blocks
        j = pid % num_x_blocks

        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(i * y_block_size, j * x_block_size),
            block_shape=(y_block_size, x_block_size),
            order=(1, 0),
        )

        grad_input_block_ptr = tl.make_block_ptr(
            grad_input_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(i * y_block_size, j * x_block_size),
            block_shape=(y_block_size, x_block_size),
            order=(1, 0),
        )

        weight_block_ptr = tl.make_block_ptr(
            weight_ptr,
            shape=x_size,
            strides=1,
            offsets=j * x_block_size,
            block_shape=x_block_size,
            order=(0,),
        )

        grad_weight_block_ptr = tl.make_block_ptr(
            grad_weight_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(i * y_block_size, j * x_block_size),
            block_shape=(y_block_size, x_block_size),
            order=(1, 0),
        )

        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(y_size, x_size),
            strides=(x_size, 1),
            offsets=(i * y_block_size, j * x_block_size),
            block_shape=(y_block_size, x_block_size),
            order=(1, 0),
        )

        input = tl.load(input_block_ptr, boundary_check=(0, 1))
        weight = tl.load(weight_block_ptr, boundary_check=(0,))
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0, 1))

        grad_input = tl.where(input > 0, 1, weight)
        grad_weight = tl.where(input > 0, 0, input)

        tl.store(grad_input_block_ptr, grad_input * grad_output, boundary_check=(0, 1))
        tl.store(grad_weight_block_ptr, grad_weight * grad_output, boundary_check=(0, 1))
