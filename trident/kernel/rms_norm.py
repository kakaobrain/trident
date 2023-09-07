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


class RMSNorm:
    @staticmethod
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
        rms_ptr: tl.tensor,
        input_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        partial_size: tl.int32,
        weight_ptr: tl.tensor,
        bias_ptr: tl.tensor,
        eps: tl.float32,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
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
        rms_block_ptr = tl.make_block_ptr(
            rms_ptr,
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
        weight_block_ptr = tl.make_block_ptr(
            weight_ptr,
            shape=(x_size,),
            strides=(1,),
            offsets=(0,),
            block_shape=(x_block_size,),
            order=(0,),
        )

        input = tl.load(input_block_ptr, boundary_check=(1,))
        partial_input = tl.where(tl.arange(0, x_block_size) < partial_size, input, 0)
        rms = tl.math.sqrt(tl.sum(partial_input * partial_input, 1) / partial_size)
        norm = input / (rms + eps)
        weight = tl.load(weight_block_ptr, boundary_check=(0,))
        output = norm * weight

        if bias_ptr is not None:
            bias_block_ptr = tl.make_block_ptr(
                bias_ptr,
                shape=(1, x_size),
                strides=(x_stride, 1),
                offsets=(0, 0),
                block_shape=(1, x_block_size),
                order=(1, 0),
            )
            bias = tl.load(bias_block_ptr, boundary_check=(1,))
            output += bias

        tl.store(rms_block_ptr, rms.to(dtype))
        tl.store(output_block_ptr, output.to(dtype), boundary_check=(1,))

    @staticmethod
    @triton.jit
    def backward(
        grad_input_ptr: tl.tensor,
        grad_weight_staging: tl.tensor,
        grad_output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        rms_ptr: tl.tensor,
        partial_size: tl.int32,
        weight_ptr: tl.tensor,
        eps: tl.float32,
        dtype: tl.constexpr,
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
        grad_weight_staging_block_ptr = tl.make_block_ptr(
            grad_weight_staging,
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
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        rms_block_ptr = tl.make_block_ptr(
            rms_ptr,
            shape=(y_size,),
            strides=(1,),
            offsets=(y_offset,),
            block_shape=(1,),
            order=(0,),
        )
        weight_block_ptr = tl.make_block_ptr(
            weight_ptr,
            shape=(x_size,),
            strides=(1,),
            offsets=(0,),
            block_shape=(x_block_size,),
            order=(0,),
        )

        grad_output = tl.load(grad_output_block_ptr, boundary_check=(1,))
        input = tl.load(input_block_ptr, boundary_check=(1,))
        rms = tl.load(rms_block_ptr)
        weight = tl.load(weight_block_ptr, boundary_check=(0,))
        grad_norm = grad_output * weight
        norm = input / (rms + eps)
        grad_weight = grad_output * norm
        tl.store(grad_weight_staging_block_ptr, grad_weight.to(dtype), boundary_check=(1,))
        grad_rms = grad_norm * -input / (rms * rms + eps)
        grad_rms = tl.where(tl.arange(0, x_block_size) < x_size, grad_rms, 0.0)
        grad_rms = tl.sum(grad_rms, 1)
        grad_mean_square = grad_rms / (2 * rms)
        grad_partial_input = 2 * input * grad_mean_square / partial_size
        grad_partial_input = tl.where(tl.arange(0, x_block_size) < partial_size, grad_partial_input, 0)
        grad_input = (grad_norm / (rms + eps)) + grad_partial_input
        tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check=(1,))
