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
    for m_block_size in [16, 32, 64]:
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


def linear_configs_for_backward_bias():
    configs = []
    for m_block_size in [32, 64, 128, 256]:
        for num_stages in [2, 3]:
            config = triton.Config(
                {
                    "m_block_size": m_block_size,
                },
                2 if m_block_size <= 64 else 4,
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
        m_size: tl.int32,
        n_size: tl.int32,
        k_size: tl.int32,
        input_batch_stride: tl.int32,
        input_m_stride: tl.int32,
        input_k_stride: tl.int32,
        weight_n_stride: tl.int32,
        weight_k_stride: tl.int32,
        use_accelerator: tl.constexpr,
        dtype: tl.constexpr,
        m_block_size: tl.constexpr,
        n_block_size: tl.constexpr,
        k_block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_m_blocks = tl.cdiv(m_size, m_block_size)
        num_n_blocks = tl.cdiv(n_size, n_block_size)
        num_blocks = num_m_blocks * num_n_blocks
        batch = pid // num_blocks
        block = pid % num_blocks
        m_block = block // num_n_blocks
        n_block = block % num_n_blocks
        m_offset = m_block * m_block_size
        n_offset = n_block * n_block_size

        output = language.Linear.forward(
            input_ptr + batch * input_batch_stride,
            weight_ptr,
            bias_ptr,
            m_size,
            n_size,
            k_size,
            input_m_stride,
            input_k_stride,
            weight_n_stride,
            weight_k_stride,
            m_offset,
            n_offset,
            use_accelerator,
            m_block_size,
            n_block_size,
            k_block_size,
            dtype,
        )
        output_block_ptr = tl.make_block_ptr(
            output_ptr + batch * m_size * n_size,
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
        m_size: tl.int32,
        n_size: tl.int32,
        k_size: tl.int32,
        input_m_stride: tl.int32,
        input_k_stride: tl.int32,
        weight_n_stride: tl.int32,
        weight_k_stride: tl.int32,
        use_accelerator: tl.constexpr,
        dtype: tl.constexpr,
        m_block_size: tl.constexpr,
        n_block_size: tl.constexpr,
        k_block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_m_blocks = tl.cdiv(m_size, m_block_size)
        num_k_blocks = tl.cdiv(k_size, k_block_size)
        num_blocks = num_m_blocks * num_k_blocks
        batch = pid // num_blocks
        block = pid % num_blocks
        m_block = block // num_k_blocks
        k_block = block % num_k_blocks
        m_offset = m_block * m_block_size
        k_offset = k_block * k_block_size

        grad_input = language.Linear.backward(
            grad_output_ptr + batch * m_size * n_size,
            weight_ptr,
            m_size,
            n_size,
            k_size,
            weight_n_stride,
            weight_k_stride,
            m_offset,
            k_offset,
            use_accelerator,
            m_block_size,
            n_block_size,
            k_block_size,
            dtype,
        )
        grad_input_block_ptr = tl.make_block_ptr(
            grad_input_ptr + batch * m_size * k_size,
            shape=(m_size, k_size),
            strides=(input_m_stride, input_k_stride),
            offsets=(m_offset, k_offset),
            block_shape=(m_block_size, k_block_size),
            order=(1, 0),
        )
        tl.store(grad_input_block_ptr, grad_input, boundary_check=(0, 1))

    @staticmethod
    @triton.autotune(configs=linear_configs(), key=["m_size", "n_size", "k_size"])
    @triton.jit
    def backward_weight(
        grad_weight_staging_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        m_size: tl.int32,
        n_size: tl.int32,
        k_size: tl.int32,
        input_batch_stride: tl.int32,
        input_m_stride: tl.int32,
        input_k_stride: tl.int32,
        use_accelerator: tl.constexpr,
        dtype: tl.constexpr,
        m_block_size: tl.constexpr,
        n_block_size: tl.constexpr,
        k_block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_n_blocks = tl.cdiv(n_size, n_block_size)
        num_k_blocks = tl.cdiv(k_size, k_block_size)
        num_blocks = num_n_blocks * num_k_blocks
        batch = pid // num_blocks
        block = pid % num_blocks
        n_block = block // num_k_blocks
        k_block = block % num_k_blocks
        n_offset = n_block * n_block_size
        k_offset = k_block * k_block_size

        grad_weight = language.Linear.backward_weight(
            grad_output_ptr + batch * m_size * n_size,
            input_ptr + batch * input_batch_stride,
            m_size,
            n_size,
            k_size,
            input_m_stride,
            input_k_stride,
            n_offset,
            k_offset,
            use_accelerator,
            m_block_size,
            n_block_size,
            k_block_size,
            dtype,
        )
        grad_weight_staging_block_ptr = tl.make_block_ptr(
            grad_weight_staging_ptr + batch * n_size * k_size,
            shape=(n_size, k_size),
            strides=(k_size, 1),
            offsets=(n_offset, k_offset),
            block_shape=(n_block_size, k_block_size),
            order=(1, 0),
        )
        tl.store(grad_weight_staging_block_ptr, grad_weight, boundary_check=(0, 1))

    @staticmethod
    @triton.autotune(configs=linear_configs_for_backward_bias(), key=["m_size", "n_size"])
    @triton.jit
    def backward_bias(
        grad_bias_staging_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        m_size: tl.int32,
        n_size: tl.int32,
        dtype: tl.constexpr,
        m_block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch = pid // n_size
        n_offset = pid % n_size
        grad_bias = language.Linear.backward_bias(
            grad_output_ptr + batch * m_size * n_size, m_size, n_size, n_offset, m_block_size, dtype
        )
        grad_bias_staging_block_ptr = tl.make_block_ptr(
            grad_bias_staging_ptr + batch * n_size,
            shape=(n_size,),
            strides=(1,),
            offsets=(n_offset,),
            block_shape=(1,),
            order=(0,),
        )
        tl.store(grad_bias_staging_block_ptr, grad_bias)
