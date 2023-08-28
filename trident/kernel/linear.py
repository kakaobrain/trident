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


def linear_configs():
    configs = []
    for m_block_size in [32, 64]:
        for n_block_size in [32, 64]:
            for k_block_size in [32, 64, 128, 256]:
                for num_stages in [2, 3]:
                    config = triton.Config(
                        {
                            "m_block_size": m_block_size,
                            "k_block_size": k_block_size,
                            "n_block_size": n_block_size,
                        },
                        2 if k_block_size <= 64 else 4,
                        num_stages,
                    )
                    configs.append(config)
    return configs


def linear_configs_for_backward_bias():
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


class Linear:
    @staticmethod
    @triton.autotune(configs=linear_configs(), key=["m_size", "n_size", "k_size"])
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
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

        output = language.Linear.forward(
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
    @triton.autotune(configs=linear_configs(), key=["m_size", "n_size", "k_size"])
    @triton.jit
    def backward(
        grad_input_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        weight_ptr: tl.tensor,
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

        grad_input = language.Linear.backward(
            grad_output_ptr,
            weight_ptr,
            m_size,
            n_size,
            k_size,
            m_offset,
            k_offset,
            use_accelerator,
            m_block_size,
            n_block_size,
            k_block_size,
            dtype,
        )
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
    @triton.autotune(configs=linear_configs(), key=["m_size", "n_size", "k_size"])
    @triton.jit
    def backward_weight(
        grad_weight_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        input_ptr: tl.tensor,
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

        grad_weight = language.Linear.backward_weight(
            grad_output_ptr,
            input_ptr,
            m_size,
            n_size,
            k_size,
            n_offset,
            k_offset,
            use_accelerator,
            m_block_size,
            n_block_size,
            k_block_size,
            dtype,
        )
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
    @triton.autotune(configs=linear_configs_for_backward_bias(), key=["m_size", "n_size"])
    @triton.jit
    def backward_bias(
        grad_bias_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        m_size: int,
        n_size: int,
        block_size: tl.constexpr,
        dtype: tl.constexpr,
    ):
        n_offset = tl.program_id(0)
        grad_bias = language.Linear.backward_bias(grad_output_ptr, m_size, n_size, n_offset, block_size, dtype)
        grad_bias_block_ptr = tl.make_block_ptr(
            grad_bias_ptr, shape=(n_size,), strides=(1,), offsets=(n_offset,), block_shape=(1,), order=(0,)
        )
        tl.store(grad_bias_block_ptr, grad_bias)
