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


def silu_configs():
    configs = []
    for x_block_size in [256, 512, 1024, 2048]:
        for num_warps in [2, 4, 8, 16]:
            config = triton.Config({"x_block_size": x_block_size}, num_warps)
            configs.append(config)
    return configs


class SiLU:
    @staticmethod
    @triton.autotune(silu_configs(), ["x_size"])
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        x_size: tl.int32,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        x_offset = tl.program_id(0) * x_block_size
        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(x_size,),
            strides=(1,),
            offsets=(x_offset,),
            block_shape=(x_block_size,),
            order=(0,),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(x_size,),
            strides=(1,),
            offsets=(x_offset,),
            block_shape=(x_block_size,),
            order=(0,),
        )
        input = tl.load(input_block_ptr, boundary_check=(0,))
        sigma = 1 / (1 + tl.math.fast_expf(-input.to(tl.float32)))
        output = input * sigma
        tl.store(output_block_ptr, output.to(dtype), boundary_check=(0,))

    @staticmethod
    @triton.autotune(silu_configs(), ["x_size"])
    @triton.jit
    def backward(
        grad_input_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        x_size: tl.int32,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        x_offset = tl.program_id(0) * x_block_size
        grad_input_block_ptr = tl.make_block_ptr(
            grad_input_ptr,
            shape=(x_size,),
            strides=(1,),
            offsets=(x_offset,),
            block_shape=(x_block_size,),
            order=(0,),
        )
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(x_size,),
            strides=(1,),
            offsets=(x_offset,),
            block_shape=(x_block_size,),
            order=(0,),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(x_size,),
            strides=(1,),
            offsets=(x_offset,),
            block_shape=(x_block_size,),
            order=(0,),
        )
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,))
        input = tl.load(input_block_ptr, boundary_check=(0,))
        sigma = 1 / (1 + tl.math.fast_expf(-input.to(tl.float32)))
        grad_input = grad_output * (sigma + input * sigma * (1 - sigma))
        tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check=(0,))
