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


def num_warps_and_stages_for_geglu(size):
    if size >= 2**15:
        num_warps = 8
        num_stages = 3
    elif size >= 2**14:
        num_warps = 4
        num_stages = 4
    else:
        num_warps = 2
        num_stages = 5
    return num_warps, num_stages


def geglu_configs():
    configs = []
    for k_block_size in [32, 64]:
        for m_block_size in [16, 64, 128]:
            for x_block_size in [32, 64, 128]:
                num_warps, num_stages = num_warps_and_stages_for_geglu(m_block_size * x_block_size)
                config = triton.Config(
                    {
                        "m_block_size": m_block_size,
                        "k_block_size": k_block_size,
                        "x_block_size": x_block_size,
                    },
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
                configs.append(config)
    return configs


class GEGLU:
    @staticmethod
    @util.autotune(geglu_configs(), ["m_size", "k_size", "x_size"])
    @triton.heuristics(
        {
            "require_m_boundary_check": lambda args: args["m_size"] % args["m_block_size"],
            "require_k_boundary_check": lambda args: args["k_size"] % args["k_block_size"],
            "require_x_boundary_check": lambda args: args["x_size"] % args["x_block_size"],
        }
    )
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
        state_gate_ptr: tl.tensor,
        input_ptr: tl.tensor,
        weight_ptr: tl.tensor,
        bias_ptr: tl.tensor,
        m_size: tl.int32,
        n_size: tl.int32,
        k_size: tl.int32,
        x_size: tl.int32,
        input_m_stride: tl.int32,
        input_k_stride: tl.int32,
        weight_n_stride: tl.int32,
        weight_k_stride: tl.int32,
        use_accelerator: tl.constexpr,
        dtype: tl.constexpr,
        m_block_size: tl.constexpr,
        k_block_size: tl.constexpr,
        x_block_size: tl.constexpr,
        require_m_boundary_check: tl.constexpr,
        require_k_boundary_check: tl.constexpr,
        require_x_boundary_check: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_m_blocks = tl.cdiv(m_size, m_block_size)
        num_x_blocks = tl.cdiv(x_size, x_block_size)
        num_blocks = num_m_blocks * num_x_blocks
        block = pid % num_blocks
        m_block = block // num_x_blocks
        x_block = block % num_x_blocks
        m_offset = m_block * m_block_size
        x_offset = x_block * x_block_size

        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(m_size, x_size),
            strides=(x_size, 1),
            offsets=(m_offset, x_offset),
            block_shape=(m_block_size, x_block_size),
            order=(1, 0),
        )
        state_block_ptr = tl.make_block_ptr(
            state_gate_ptr,
            shape=(m_size, n_size),
            strides=(n_size, 1),
            offsets=(m_offset, x_offset),
            block_shape=(m_block_size, x_block_size),
            order=(1, 0),
        )
        gate_block_ptr = tl.make_block_ptr(
            state_gate_ptr,
            shape=(m_size, n_size),
            strides=(n_size, 1),
            offsets=(m_offset, x_offset + x_size),
            block_shape=(m_block_size, x_block_size),
            order=(1, 0),
        )

        state = language.Linear.forward(
            input_ptr,
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
            x_offset,
            use_accelerator,
            m_block_size,
            x_block_size,
            k_block_size,
            require_m_boundary_check,
            require_x_boundary_check,
            require_k_boundary_check,
            dtype,
        )
        gate = language.Linear.forward(
            input_ptr,
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
            x_offset + x_size,
            use_accelerator,
            m_block_size,
            x_block_size,
            k_block_size,
            require_m_boundary_check,
            require_x_boundary_check,
            require_k_boundary_check,
            dtype,
        )
        output = state * language.math.GELU.forward(gate)

        if require_m_boundary_check | require_x_boundary_check:
            tl.store(output_block_ptr, output.to(dtype), boundary_check=(0, 1))
            tl.store(state_block_ptr, state.to(dtype), boundary_check=(0, 1))
            tl.store(gate_block_ptr, gate.to(dtype), boundary_check=(0, 1))
        else:
            tl.store(output_block_ptr, output.to(dtype))
            tl.store(state_block_ptr, state.to(dtype))
            tl.store(gate_block_ptr, gate.to(dtype))

    @staticmethod
    @triton.heuristics(
        {
            "require_x_boundary_check": lambda args: args["x_size"] % args["x_block_size"],
        }
    )
    @triton.jit
    def backward(
        grad_state_gate_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        state_gate_ptr: tl.tensor,
        m_size: tl.int32,
        n_size: tl.int32,
        x_size: tl.int32,
        dtype: tl.constexpr,
        x_block_size: tl.constexpr,
        require_x_boundary_check: tl.constexpr,
    ):
        pid = tl.program_id(0)
        m_offset = pid % m_size

        grad_state_block_ptr = tl.make_block_ptr(
            grad_state_gate_ptr,
            shape=(m_size, n_size),
            strides=(n_size, 1),
            offsets=(m_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        grad_gate_block_ptr = tl.make_block_ptr(
            grad_state_gate_ptr,
            shape=(m_size, n_size),
            strides=(n_size, 1),
            offsets=(m_offset, x_size),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        grad_output_block_ptr = tl.make_block_ptr(
            grad_output_ptr,
            shape=(m_size, x_size),
            strides=(x_size, 1),
            offsets=(m_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        state_block_ptr = tl.make_block_ptr(
            state_gate_ptr,
            shape=(m_size, n_size),
            strides=(n_size, 1),
            offsets=(m_offset, 0),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )
        gate_block_ptr = tl.make_block_ptr(
            state_gate_ptr,
            shape=(m_size, n_size),
            strides=(n_size, 1),
            offsets=(m_offset, x_size),
            block_shape=(1, x_block_size),
            order=(1, 0),
        )

        if require_x_boundary_check:
            grad_output = tl.load(grad_output_block_ptr, boundary_check=(1,))
            state = tl.load(state_block_ptr, boundary_check=(1,))
            gate = tl.load(gate_block_ptr, boundary_check=(1,))
        else:
            grad_output = tl.load(grad_output_block_ptr)
            state = tl.load(state_block_ptr)
            gate = tl.load(gate_block_ptr)

        grad_state = grad_output * language.math.GELU.forward(gate)
        grad_gate = language.math.GELU.backward(grad_output * state, gate)

        if require_x_boundary_check:
            tl.store(grad_state_block_ptr, grad_state.to(dtype), boundary_check=(1,))
            tl.store(grad_gate_block_ptr, grad_gate.to(dtype), boundary_check=(1,))
        else:
            tl.store(grad_state_block_ptr, grad_state.to(dtype))
            tl.store(grad_gate_block_ptr, grad_gate.to(dtype))
