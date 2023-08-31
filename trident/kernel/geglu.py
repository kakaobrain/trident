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


def geglu_configs():
    configs = []
    for m_block_size in [32, 64]:
        for n_block_size in [32, 64]:
            for k_block_size in [32, 64, 128, 256]:
                for num_stages in [2, 3]:
                    config = triton.Config(
                        {
                            "m_block_size": m_block_size,
                            "n_block_size": n_block_size,
                            "k_block_size": k_block_size,
                        },
                        2 if k_block_size <= 64 else 4,
                        num_stages,
                    )
                    configs.append(config)
    return configs


def geglu_configs_for_backward_bias():
    configs = []
    for block_size in [32, 64, 128, 256]:
        for num_stages in [2, 3]:
            config = triton.Config(
                {
                    "block_size": block_size,
                },
                2 if block_size <= 64 else 4,
                num_stages,
            )
            configs.append(config)
    return configs


class GEGLU:
    @staticmethod
    @triton.autotune(configs=geglu_configs(), key=["m_size", "n_size", "k_size"])
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
        linear_ptr: tl.tensor,
        input_ptr: tl.tensor,
        weight_ptr: tl.tensor,
        bias_ptr: tl.tensor,
        m_size: int,
        n_size: int,
        k_size: int,
        use_accelerator: tl.constexpr,
        m_block_size: tl.constexpr,
        n_block_size: tl.constexpr,
        k_block_size: tl.constexpr,
        dtype: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_n_blocks = tl.cdiv(n_size, n_block_size)
        m_block = pid // num_n_blocks
        n_block = pid % num_n_blocks
        m_offset = m_block * m_block_size
        n_offset = n_block * n_block_size

        linear = language.Linear.forward(
            input_ptr,
            weight_ptr,
            bias_ptr,
            m_size,
            n_size,
            k_size,
            m_offset,
            n_offset,
            use_accelerator,
            m_block_size,
            n_block_size,
            k_block_size,
            dtype,
        )
        linear_block_ptr = tl.make_block_ptr(
            linear_ptr,
            shape=(m_size, n_size),
            strides=(n_size, 1),
            offsets=(m_offset, n_offset),
            block_shape=(m_block_size, n_block_size),
            order=(1, 0),
        )
        tl.store(linear_block_ptr, linear, boundary_check=(0, 1))

        output = language.math.GELU.forward(linear)
        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(m_size, n_size),
            strides=(n_size, 1),
            offsets=(m_offset, n_offset),
            block_shape=(m_block_size, n_block_size),
            order=(1, 0),
        )
        tl.store(output_block_ptr, output, boundary_check=(0, 1))

    @staticmethod
    @triton.autotune(configs=geglu_configs(), key=["m_size", "n_size", "k_size"])
    @triton.jit
    def backward(
        grad_input_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        weight_ptr: tl.tensor,
        linear_ptr: tl.tensor,
        m_size: int,
        n_size: int,
        k_size: int,
        use_accelerator: tl.constexpr,
        m_block_size: tl.constexpr,
        n_block_size: tl.constexpr,
        k_block_size: tl.constexpr,
        dtype: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_k_blocks = tl.cdiv(k_size, k_block_size)
        m_block = pid // num_k_blocks
        k_block = pid % num_k_blocks
        m_offset = m_block * m_block_size
        k_offset = k_block * k_block_size

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
        linear_block_ptr = tl.make_block_ptr(
            linear_ptr,
            shape=(m_size, n_size),
            strides=(n_size, 1),
            offsets=(m_offset, 0),
            block_shape=(m_block_size, n_block_size),
            order=(1, 0),
        )
        grad_input = tl.zeros((m_block_size, k_block_size), dtype)

        for n_offset in range(0, n_size, n_block_size):
            grad_output = tl.load(grad_output_block_ptr, boundary_check=(0, 1), padding_option="zero")
            linear = tl.load(linear_block_ptr, boundary_check=(0, 1), padding_option="zero")
            grad_output = language.math.GELU.backward(grad_output, linear)
            weight = tl.load(weight_block_ptr, boundary_check=(0, 1), padding_option="zero")
            grad_input += tl.dot(grad_output, weight, use_accelerator)
            grad_output_block_ptr = tl.advance(grad_output_block_ptr, (0, n_block_size))
            linear_block_ptr = tl.advance(linear_block_ptr, (0, n_block_size))
            weight_block_ptr = tl.advance(weight_block_ptr, (n_block_size, 0))

        grad_input_block_ptr = tl.make_block_ptr(
            grad_input_ptr,
            shape=(m_size, k_size),
            strides=(k_size, 1),
            offsets=(m_offset, k_offset),
            block_shape=(m_block_size, k_block_size),
            order=(1, 0),
        )
        tl.store(grad_input_block_ptr, grad_input, boundary_check=(0, 1))

    @staticmethod
    @triton.autotune(configs=geglu_configs(), key=["m_size", "n_size", "k_size"])
    @triton.jit
    def backward_weight(
        grad_weight_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        linear_ptr: tl.tensor,
        m_size: int,
        n_size: int,
        k_size: int,
        use_accelerator: tl.constexpr,
        m_block_size: tl.constexpr,
        n_block_size: tl.constexpr,
        k_block_size: tl.constexpr,
        dtype: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_k_blocks = tl.cdiv(k_size, k_block_size)
        n_block = pid // num_k_blocks
        k_block = pid % num_k_blocks
        n_offset = n_block * n_block_size
        k_offset = k_block * k_block_size

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
        linear_block_ptr = tl.make_block_ptr(
            linear_ptr,
            shape=(n_size, m_size),
            strides=(1, n_size),
            offsets=(n_offset, 0),
            block_shape=(n_block_size, m_block_size),
            order=(0, 1),
        )
        grad_weight = tl.zeros((n_block_size, k_block_size), dtype)

        for m_offset in range(0, m_size, m_block_size):
            grad_output = tl.load(grad_output_block_ptr, boundary_check=(0, 1), padding_option="zero")
            linear = tl.load(linear_block_ptr, boundary_check=(0, 1), padding_option="zero")
            grad_output = language.math.GELU.backward(grad_output, linear)
            input = tl.load(input_block_ptr, boundary_check=(0, 1), padding_option="zero")
            grad_weight += tl.dot(grad_output, input, use_accelerator)
            grad_output_block_ptr = tl.advance(grad_output_block_ptr, (0, m_block_size))
            linear_block_ptr = tl.advance(linear_block_ptr, (0, m_block_size))
            input_block_ptr = tl.advance(input_block_ptr, (m_block_size, 0))

        grad_weight_block_ptr = tl.make_block_ptr(
            grad_weight_ptr,
            shape=(n_size, k_size),
            strides=(k_size, 1),
            offsets=(n_offset, k_offset),
            block_shape=(n_block_size, k_block_size),
            order=(1, 0),
        )
        tl.store(grad_weight_block_ptr, grad_weight, boundary_check=(0, 1))

    @staticmethod
    @triton.autotune(configs=geglu_configs_for_backward_bias(), key=["m_size", "n_size"])
    @triton.jit
    def backward_bias(
        grad_bias_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        linear_ptr: tl.tensor,
        m_size: int,
        n_size: int,
        block_size: tl.constexpr,
        dtype: tl.constexpr,
    ):
        offset = tl.program_id(0)
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(n_size, m_size),
            strides=(1, n_size),
            offsets=(offset, 0),
            block_shape=(1, block_size),
            order=(0, 1),
        )
        linear_block_ptr = tl.make_block_ptr(
            linear_ptr,
            shape=(n_size, m_size),
            strides=(1, n_size),
            offsets=(offset, 0),
            block_shape=(1, block_size),
            order=(0, 1),
        )
        sum = tl.zeros((1, block_size), dtype)

        for m_offset in range(0, m_size, block_size):
            grad_output = tl.load(grad_output_block_ptr, boundary_check=(1,), padding_option="zero")
            linear = tl.load(linear_block_ptr, boundary_check=(0, 1), padding_option="zero")
            grad_output = language.math.GELU.backward(grad_output, linear)
            sum += grad_output
            grad_output_block_ptr = tl.advance(grad_output_block_ptr, (0, block_size))
            linear_block_ptr = tl.advance(linear_block_ptr, (0, block_size))

        grad_bias = tl.sum(sum, 1)
        grad_bias_block_ptr = tl.make_block_ptr(
            grad_bias_ptr, shape=(n_size,), strides=(1,), offsets=(offset,), block_shape=(1,), order=(0,)
        )
        tl.store(grad_bias_block_ptr, grad_bias)
