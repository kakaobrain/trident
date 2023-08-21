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


class Linear:
    @staticmethod
    @triton.jit
    def forward(
        input_ptr: tl.tensor,
        weight_ptr: tl.tensor,
        bias_ptr: tl.tensor,
        m_size: int,
        n_size: int,
        k_size: int,
        m_offset: int,
        n_offset: int,
        use_accelerator: tl.constexpr,
        m_block_size: tl.constexpr,
        n_block_size: tl.constexpr,
        k_block_size: tl.constexpr,
        dtype: tl.constexpr,
    ):
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(m_size, k_size),
            strides=(k_size, 1),
            offsets=(m_offset, 0),
            block_shape=(m_block_size, k_block_size),
            order=(1, 0),
        )
        weight_block_ptr = tl.make_block_ptr(
            weight_ptr,
            shape=(k_size, n_size),
            strides=(1, k_size),
            offsets=(0, n_offset),
            block_shape=(k_block_size, n_block_size),
            order=(0, 1),
        )
        output = tl.zeros((m_block_size, n_block_size), dtype)

        for k_offset in range(0, k_size, k_block_size):
            input = tl.load(input_block_ptr, boundary_check=(0, 1), padding_option="zero")
            weight = tl.load(weight_block_ptr, boundary_check=(0, 1), padding_option="zero")
            output += tl.dot(input, weight, use_accelerator).to(dtype)
            input_block_ptr = tl.advance(input_block_ptr, (0, k_block_size))
            weight_block_ptr = tl.advance(weight_block_ptr, (k_block_size, 0))

        if bias_ptr is not None:
            bias_block_ptr = tl.make_block_ptr(
                bias_ptr,
                shape=(n_size,),
                strides=(1,),
                offsets=(n_offset,),
                block_shape=(n_block_size,),
                order=(0,),
            )
            bias = tl.load(bias_block_ptr, boundary_check=(0,), padding_option="zero")
            output += bias

        return output

    @staticmethod
    @triton.jit
    def backward(
        grad_output_ptr: tl.tensor,
        weight_ptr: tl.tensor,
        m_size: int,
        n_size: int,
        k_size: int,
        m_offset: int,
        k_offset: int,
        use_accelerator: tl.constexpr,
        m_block_size: tl.constexpr,
        n_block_size: tl.constexpr,
        k_block_size: tl.constexpr,
        dtype: tl.constexpr,
    ):
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(m_size, n_size),
            strides=(n_size, 1),
            offsets=(m_offset, 0),
            block_shape=(m_block_size, n_block_size),
            order=(1, 0),
        )
        weight_block_ptr = tl.make_block_ptr(
            weight_ptr,
            shape=(n_size, k_size),
            strides=(k_size, 1),
            offsets=(0, k_offset),
            block_shape=(n_block_size, k_block_size),
            order=(1, 0),
        )
        grad_input = tl.zeros((m_block_size, k_block_size), dtype)

        for n_offset in range(0, n_size, n_block_size):
            grad_output = tl.load(grad_output_block_ptr, boundary_check=(0, 1), padding_option="zero")
            weight = tl.load(weight_block_ptr, boundary_check=(0, 1), padding_option="zero")
            grad_input += tl.dot(grad_output, weight, use_accelerator)
            grad_output_block_ptr = tl.advance(grad_output_block_ptr, (0, n_block_size))
            weight_block_ptr = tl.advance(weight_block_ptr, (n_block_size, 0))

        return grad_input

    @staticmethod
    @triton.jit
    def backward_weight(
        grad_output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        m_size: int,
        n_size: int,
        k_size: int,
        n_offset: int,
        k_offset: int,
        use_accelerator: tl.constexpr,
        m_block_size: tl.constexpr,
        n_block_size: tl.constexpr,
        k_block_size: tl.constexpr,
        dtype: tl.constexpr,
    ):
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(n_size, m_size),
            strides=(1, n_size),
            offsets=(n_offset, 0),
            block_shape=(n_block_size, m_block_size),
            order=(0, 1),
        )
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(m_size, k_size),
            strides=(k_size, 1),
            offsets=(0, k_offset),
            block_shape=(m_block_size, k_block_size),
            order=(1, 0),
        )
        grad_weight = tl.zeros((n_block_size, k_block_size), dtype)

        for m_offset in range(0, m_size, m_block_size):
            grad_output = tl.load(grad_output_block_ptr, boundary_check=(0, 1), padding_option="zero")
            input = tl.load(input_block_ptr, boundary_check=(0, 1), padding_option="zero")
            grad_weight += tl.dot(grad_output, input, use_accelerator)
            grad_output_block_ptr = tl.advance(grad_output_block_ptr, (0, m_block_size))
            input_block_ptr = tl.advance(input_block_ptr, (m_block_size, 0))

        return grad_weight

    @staticmethod
    @triton.jit
    def backward_bias(
        grad_output_ptr: tl.tensor,
        m_size: int,
        n_size: int,
        n_offset: int,
        m_block_size: tl.constexpr,
        dtype: tl.constexpr,
    ):
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(n_size, m_size),
            strides=(1, n_size),
            offsets=(n_offset, 0),
            block_shape=(1, m_block_size),
            order=(0, 1),
        )
        grad_bias = tl.zeros((1, m_block_size), dtype)

        for m_offset in range(0, m_size, m_block_size):
            grad_output = tl.load(grad_output_block_ptr, boundary_check=(1,), padding_option="zero")
            grad_bias += grad_output
            grad_output_block_ptr = tl.advance(grad_output_block_ptr, (0, m_block_size))

        return tl.sum(grad_bias, 1)
