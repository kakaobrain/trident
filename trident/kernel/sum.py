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


def sum_configs():
    configs = []
    for x_block_size in [512, 1024, 2048]:
        for num_stages in [4, 5]:
            config = triton.Config({"x_block_size": x_block_size}, 2 if x_block_size <= 1024 else 4, num_stages)
            configs.append(config)
    return configs


class Sum:
    @staticmethod
    @util.autotune(sum_configs(), ["x_size"])
    @triton.heuristics({"require_x_boundary_check": lambda args: args["x_size"] % args["x_block_size"]})
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
        require_x_boundary_check: tl.constexpr,
    ):
        y_offset = tl.program_id(0)

        output = language.Sum.forward(
            input_ptr, y_size, x_size, y_stride, x_stride, y_offset, dtype, x_block_size, require_x_boundary_check
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
    @util.autotune(sum_configs(), ["x_size"])
    @triton.heuristics({"require_x_boundary_check": lambda args: args["x_size"] % args["x_block_size"]})
    @triton.jit
    def backward(
        grad_input_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        x_block_size: tl.constexpr,
        require_x_boundary_check: tl.constexpr,
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
        grad_input = language.Sum.backward(grad_output_ptr, y_size, y_offset, x_block_size)

        for x_offset in range(0, x_size, x_block_size):
            if require_x_boundary_check:
                tl.store(grad_input_block_ptr, grad_input, boundary_check=(1,))
            else:
                tl.store(grad_input_block_ptr, grad_input)

            grad_input_block_ptr = tl.advance(grad_input_block_ptr, (0, x_block_size))
