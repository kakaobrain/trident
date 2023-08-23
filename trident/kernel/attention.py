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


class Attention:
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
        log_sum_exp_ptr: tl.tensor,
        query_ptr: tl.tensor,
        key_ptr: tl.tensor,
        value_ptr: tl.tensor,
        softmax_scale: tl.constexpr,
        query_head_stride: int,
        query_sequence_stride: int,
        query_embedding_stride: int,
        key_sequence_stride: int,
        key_embedding_stride: int,
        value_sequence_stride: int,
        value_embedding_stride: int,
        output_sequence_stride: int,
        output_embedding_stride: int,
        sequence_size: tl.constexpr,
        m_block_size: tl.constexpr,
        n_block_size: tl.constexpr,
        embedding_size: tl.constexpr,
        is_causal: tl.constexpr,
        use_accelerator: tl.constexpr,
        dtype: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        offset_hz = tl.program_id(1)
        offset = offset_hz * query_head_stride
        query_block_ptr = tl.make_block_ptr(
            query_ptr + offset,
            shape=(sequence_size, embedding_size),
            strides=(query_sequence_stride, query_embedding_stride),
            offsets=(start_m * m_block_size, 0),
            block_shape=(m_block_size, embedding_size),
            order=(1, 0),
        )
        key_block_ptr = tl.make_block_ptr(
            key_ptr + offset,
            shape=(embedding_size, sequence_size),
            strides=(key_embedding_stride, key_sequence_stride),
            offsets=(0, 0),
            block_shape=(embedding_size, n_block_size),
            order=(0, 1),
        )
        value_block_ptr = tl.make_block_ptr(
            value_ptr + offset,
            shape=(sequence_size, embedding_size),
            strides=(value_sequence_stride, value_embedding_stride),
            offsets=(0, 0),
            block_shape=(n_block_size, embedding_size),
            order=(1, 0),
        )
        offsets_m = start_m * m_block_size + tl.arange(0, n_block_size)
        offsets_n = tl.arange(0, n_block_size)

        max_i = tl.zeros([m_block_size], dtype=tl.float32) - float("inf")
        sum_exp_i = tl.zeros([m_block_size], dtype=tl.float32)
        accumulation = tl.zeros([m_block_size, embedding_size], dtype=tl.float32)
        qk_scale = softmax_scale * tl.math.log2(language.e)
        q = tl.load(query_block_ptr, boundary_check=(0, 1))
        q = (q * qk_scale).to(tl.float32)

        highest = (start_m + 1) * m_block_size if is_causal else sequence_size
        for start_n in range(0, highest, n_block_size):
            k = tl.load(key_block_ptr, boundary_check=(0, 1)).to(tl.float32)
            v = tl.load(value_block_ptr, boundary_check=(0, 1)).to(tl.float32)
            qk = tl.zeros([m_block_size, n_block_size], dtype=tl.float32)
            if is_causal:
                qk = tl.where(offsets_m[:, None] >= (start_n + offsets_n[None, :]), qk, float("-inf"))
            qk += tl.dot(q, k, allow_tf32=use_accelerator)
            max_i_new = tl.maximum(max_i, tl.max(qk, 1))
            alpha = tl.math.exp2(max_i - max_i_new)
            p = tl.math.exp2(qk - max_i_new[:, None])
            accumulation_scale = sum_exp_i * 0 + alpha  # workaround some compiler bug
            accumulation *= accumulation_scale[:, None]
            accumulation += tl.dot(p.to(tl.float32), v, allow_tf32=use_accelerator)
            sum_exp_i = sum_exp_i * alpha + tl.sum(p, 1)
            max_i = max_i_new
            key_block_ptr = tl.advance(key_block_ptr, (0, n_block_size))
            value_block_ptr = tl.advance(value_block_ptr, (n_block_size, 0))

        accumulation = accumulation / sum_exp_i[:, None]
        log_sum_exp_ptrs = log_sum_exp_ptr + offset_hz * sequence_size + offsets_m
        tl.store(log_sum_exp_ptrs, max_i + tl.math.log2(sum_exp_i))
        output_block_ptr = tl.make_block_ptr(
            output_ptr + offset,
            shape=(sequence_size, embedding_size),
            strides=(output_sequence_stride, output_embedding_stride),
            offsets=(start_m * m_block_size, 0),
            block_shape=(m_block_size, embedding_size),
            order=(1, 0),
        )
        tl.store(output_block_ptr, accumulation.to(dtype))

    @triton.jit
    def backward(
        grad_query_ptr: tl.tensor,
        grad_key_ptr: tl.tensor,
        grad_value_ptr: tl.tensor,
        query_ptr: tl.tensor,
        key_ptr: tl.tensor,
        value_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        log_sum_exp_ptr: tl.tensor,
        delta_ptr: tl.tensor,
        query_z_stride: int,
        query_h_stride: int,
        query_m_stride: int,
        query_k_stride: int,
        key_n_stride: int,
        key_k_stride: int,
        num_head: tl.constexpr,
        sequence_size: tl.constexpr,
        num_block: tl.constexpr,
        softmax_scale: tl.constexpr,
        m_block_size: tl.constexpr,
        n_block_size: tl.constexpr,
        embedding_size: tl.constexpr,
        is_causal: tl.constexpr,
        use_accelerator: tl.constexpr,
        dtype: tl.constexpr,
    ):
        offset_hz = tl.program_id(0)
        offset_z = offset_hz // num_head
        offset_h = offset_hz % num_head
        qk_scale = softmax_scale * tl.math.log2(language.e)
        query_ptr += offset_z * query_z_stride + offset_h * query_h_stride
        key_ptr += offset_z * query_z_stride + offset_h * query_h_stride
        value_ptr += offset_z * query_z_stride + offset_h * query_h_stride
        grad_output_ptr += offset_z * query_z_stride + offset_h * query_h_stride
        grad_query_ptr += offset_z * query_z_stride + offset_h * query_h_stride
        grad_key_ptr += offset_z * query_z_stride + offset_h * query_h_stride
        grad_value_ptr += offset_z * query_z_stride + offset_h * query_h_stride

        for start_n in range(0, num_block):
            if is_causal:
                lo = start_n * m_block_size
            else:
                lo = 0

            offsets_qm = lo + tl.arange(0, m_block_size)
            offsets_n = start_n * m_block_size + tl.arange(0, m_block_size)
            offsets_m = tl.arange(0, n_block_size)
            offsets_k = tl.arange(0, embedding_size)
            query_ptrs = query_ptr + (offsets_qm[:, None] * query_m_stride + offsets_k[None, :] * query_k_stride)
            key_ptrs = key_ptr + (offsets_n[:, None] * key_n_stride + offsets_k[None, :] * key_k_stride)
            value_ptrs = value_ptr + (offsets_n[:, None] * query_m_stride + offsets_k[None, :] * query_k_stride)
            grad_output_ptrs = grad_output_ptr + (
                offsets_qm[:, None] * query_m_stride + offsets_k[None, :] * query_k_stride
            )
            grad_query_ptrs = grad_query_ptr + (
                offsets_qm[:, None] * query_m_stride + offsets_k[None, :] * query_k_stride
            )
            delta_ptrs = delta_ptr + offset_hz * sequence_size
            log_sum_exp_ptrs = log_sum_exp_ptr + offset_hz * sequence_size
            grad_value = tl.zeros([m_block_size, embedding_size], dtype=tl.float32)
            grad_key = tl.zeros([m_block_size, embedding_size], dtype=tl.float32)
            k = tl.load(key_ptrs)
            v = tl.load(value_ptrs)

            for start_m in range(lo, num_block * m_block_size, m_block_size):
                offsets_m_curr = start_m + offsets_m
                q = tl.load(query_ptrs)
                if is_causal:
                    qk = tl.where(offsets_m_curr[:, None] >= (offsets_n[None, :]), float(0.0), float("-inf"))
                else:
                    qk = tl.zeros([m_block_size, n_block_size], dtype=tl.float32)
                qk += tl.dot(q, tl.trans(k), allow_tf32=use_accelerator)
                qk *= qk_scale
                l_i = tl.load(log_sum_exp_ptrs + offsets_m_curr)
                p = tl.math.exp2(qk - l_i[:, None])
                grad_output = tl.load(grad_output_ptrs)
                grad_value += tl.dot(tl.trans(p), grad_output, allow_tf32=use_accelerator)
                delta_i = tl.load(delta_ptrs + offsets_m_curr)
                grad_p = tl.zeros([m_block_size, n_block_size], dtype=tl.float32) - delta_i[:, None]
                grad_p += tl.dot(grad_output, tl.trans(v), allow_tf32=use_accelerator)
                grad_s = p * grad_p * softmax_scale
                grad_key += tl.dot(tl.trans(grad_s), q, allow_tf32=use_accelerator)
                grad_query = tl.load(grad_query_ptrs)
                grad_query += tl.dot(grad_s, k, allow_tf32=use_accelerator)
                tl.store(grad_query_ptrs, grad_query.to(dtype))
                grad_query_ptrs += m_block_size * query_m_stride
                query_ptrs += m_block_size * query_m_stride
                grad_output_ptrs += m_block_size * query_m_stride

            grad_value_ptrs = grad_value_ptr + (
                offsets_n[:, None] * query_m_stride + offsets_k[None, :] * query_k_stride
            )
            grad_key_ptrs = grad_key_ptr + (offsets_n[:, None] * key_n_stride + offsets_k[None, :] * key_k_stride)
            tl.store(grad_value_ptrs, grad_value.to(dtype))
            tl.store(grad_key_ptrs, grad_key.to(dtype))
