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


class MaskedSoftmax:
    @staticmethod
    @triton.heuristics({"require_x_boundary_check": lambda args: args["x_size"] % args["x_block_size"]})
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        mask_ptr: tl.tensor,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
        require_x_boundary_check: tl.constexpr,
    ):
        y_offset = tl.program_id(0)

        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        mask_block_ptr = tl.make_block_ptr(
            mask_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )

        if require_x_boundary_check:
            input = tl.load(input_block_ptr, boundary_check=(1,))
            mask = tl.load(mask_block_ptr, boundary_check=(1,))
            condition = tl.arange(0, x_block_size) < x_size
            mask = tl.where(condition, mask, 1)
        else:
            input = tl.load(input_block_ptr)
            mask = tl.load(mask_block_ptr)

        input = tl.where(mask > language.eps, float("-inf"), input)
        max = tl.max(input, 1)
        numerator = tl.math.fast_expf(input - max)
        output = numerator / tl.sum(numerator)

        if require_x_boundary_check:
            tl.store(output_block_ptr, output.to(dtype), boundary_check=(1,))
        else:
            tl.store(output_block_ptr, output.to(dtype))

    @staticmethod
    @triton.heuristics({"require_x_boundary_check": lambda args: args["x_size"] % args["x_block_size"]})
    @triton.jit
    def backward(
        grad_input_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        output_ptr: tl.tensor,
        delta_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        dtype: tl.constexpr,
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
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        delta_block_ptr = tl.make_block_ptr(
            delta_ptr,
            shape=(y_size,),
            strides=(1,),
            offsets=(y_offset,),
            block_shape=(1,),
            order=(0,),
        )

        if require_x_boundary_check:
            output = tl.load(output_block_ptr, boundary_check=(1,))
            grad_output = tl.load(grad_output_block_ptr, boundary_check=(1,))
        else:
            output = tl.load(output_block_ptr)
            grad_output = tl.load(grad_output_block_ptr)

        delta = tl.load(delta_block_ptr)
        grad_input = output * (grad_output - delta)

        if require_x_boundary_check:
            tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check=(1,))
        else:
            tl.store(grad_input_block_ptr, grad_input.to(dtype))

    @staticmethod
    @triton.heuristics({"require_x_boundary_check": lambda args: args["x_size"] % args["x_block_size"]})
    @triton.jit
    def backward_delta(
        delta_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        output_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
        require_x_boundary_check: tl.constexpr,
    ):
        y_offset = tl.program_id(0)

        delta_block_ptr = tl.make_block_ptr(
            delta_ptr,
            shape=(y_size,),
            strides=(1,),
            offsets=(y_offset,),
            block_shape=(1,),
            order=(0,),
        )
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )

        if require_x_boundary_check:
            grad_output = tl.load(grad_output_block_ptr, boundary_check=(1,), padding_option="zero")
            output = tl.load(output_block_ptr, boundary_check=(1,))
        else:
            grad_output = tl.load(grad_output_block_ptr)
            output = tl.load(output_block_ptr)

        delta = tl.sum(grad_output * output, 1)
        tl.store(delta_block_ptr, delta.to(dtype))
