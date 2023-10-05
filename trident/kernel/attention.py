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


def attention_configs():
    configs = []
    for y_block_size in [16, 32, 64]:
        for num_warps in [2, 4, 8]:
            configs.append(triton.Config({"y_block_size": y_block_size}, num_warps))
    return configs


class Attention:
    @staticmethod
    @util.autotune(attention_configs(), ["y_size"])
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
        log2sum_ptr: tl.tensor,
        query_ptr: tl.tensor,
        key_ptr: tl.tensor,
        value_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        head_stride: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        mask_ptr: tl.tensor,
        mask_head_stride: tl.int32,
        mask_y_stride: tl.int32,
        mask_x_stride: tl.int32,
        dropout_p: tl.float32,
        seed: tl.int32,
        is_causal: tl.constexpr,
        softmax_scale: tl.float32,
        use_accelerator: tl.constexpr,
        dtype: tl.constexpr,
        y_block_size: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_y_blocks = tl.cdiv(y_size, y_block_size)
        head = pid // num_y_blocks
        y_block = pid % num_y_blocks
        head_offset = head * head_stride
        y_offset = y_block * y_block_size

        output_block_ptr = tl.make_block_ptr(
            output_ptr + head_offset,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(y_block_size, x_block_size),
            order=(1, 0),
        )
        log2sum_block_ptr = tl.make_block_ptr(
            log2sum_ptr + head * y_size,
            shape=(y_size,),
            strides=(1,),
            offsets=(y_offset,),
            block_shape=(y_block_size,),
            order=(0,),
        )
        query_block_ptr = tl.make_block_ptr(
            query_ptr + head_offset,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(y_offset, 0),
            block_shape=(y_block_size, x_block_size),
            order=(1, 0),
        )
        key_block_ptr = tl.make_block_ptr(
            key_ptr + head_offset,
            shape=(x_size, y_size),
            strides=(x_stride, y_stride),
            offsets=(0, 0),
            block_shape=(x_block_size, y_block_size),
            order=(0, 1),
        )
        value_block_ptr = tl.make_block_ptr(
            value_ptr + head_offset,
            shape=(y_size, x_size),
            strides=(y_stride, x_stride),
            offsets=(0, 0),
            block_shape=(y_block_size, x_block_size),
            order=(1, 0),
        )

        if mask_ptr is not None:
            mask_block_ptr = tl.make_block_ptr(
                mask_ptr + head * mask_head_stride,
                shape=(y_size, y_size),
                strides=(mask_y_stride, mask_x_stride),
                offsets=(y_offset, 0),
                block_shape=(y_block_size, y_block_size),
                order=(1, 0),
            )

        query = tl.load(query_block_ptr)
        score_scale = (softmax_scale * language.log2e).to(dtype)
        query *= score_scale
        max = tl.full((y_block_size,), float("-inf"), tl.float32)
        sum = tl.zeros((y_block_size,), tl.float32)
        output = tl.zeros((y_block_size, x_block_size), dtype)
        m_offsets = tl.arange(0, y_block_size) + y_offset

        if is_causal:
            n_size = y_offset + y_block_size
        else:
            n_size = y_size

        for n_offset in range(0, n_size, y_block_size):
            score = tl.zeros((y_block_size, y_block_size), dtype)

            if is_causal:
                n_offsets = tl.arange(0, y_block_size) + n_offset
                condition = m_offsets[:, None] >= n_offsets[None, :]
                score = tl.where(condition, score, float("-inf"))
            elif mask_ptr is not None:
                mask = tl.load(mask_block_ptr)
                mask *= language.log2e
                score += mask

            key = tl.load(key_block_ptr)
            score += language.dot(query, key, use_accelerator, dtype)
            peak = tl.maximum(max, tl.max(score, 1))
            alpha = tl.math.exp2(max - peak)
            beta = tl.math.exp2(score - peak[:, None])
            sum = sum * alpha + tl.sum(beta, 1)
            max = peak
            output *= alpha[:, None].to(dtype)
            value = tl.load(value_block_ptr)
            output += language.dot(beta.to(dtype), value, use_accelerator, dtype)
            key_block_ptr = tl.advance(key_block_ptr, (0, y_block_size))
            value_block_ptr = tl.advance(value_block_ptr, (y_block_size, 0))

            if mask_ptr is not None:
                mask_block_ptr = tl.advance(mask_block_ptr, (0, y_block_size))

        output /= sum[:, None].to(dtype)

        if dropout_p > language.eps:
            dropout_mask = tl.rand(seed, tl.arange(0, x_block_size) + y_offset) > dropout_p
            dropout_scale = 1.0 - dropout_p + language.eps
            output = tl.where(dropout_mask, output / dropout_scale, 0.0)

        tl.store(output_block_ptr, output.to(dtype))
        log2sum = max + tl.math.log2(sum)
        tl.store(log2sum_block_ptr, log2sum.to(dtype))

    @staticmethod
    @triton.jit
    def backward(
        grad_query_ptr: tl.tensor,
        grad_key_ptr: tl.tensor,
        grad_value_ptr: tl.tensor,
        grad_mask_ptr: tl.tensor,
        grad_output_ptr: tl.tensor,
        query_ptr: tl.tensor,
        key_ptr: tl.tensor,
        value_ptr: tl.tensor,
        y_size: tl.int32,
        x_size: tl.int32,
        head_stride: tl.int32,
        y_stride: tl.int32,
        x_stride: tl.int32,
        mask_ptr: tl.tensor,
        mask_head_stride: tl.int32,
        mask_y_stride: tl.int32,
        mask_x_stride: tl.int32,
        output_ptr: tl.tensor,
        log2sum_ptr: tl.tensor,
        delta_ptr: tl.tensor,
        dropout_p: tl.float32,
        is_causal: tl.constexpr,
        softmax_scale: tl.float32,
        use_accelerator: tl.constexpr,
        dtype: tl.constexpr,
        y_block_size: tl.constexpr,
        x_block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_y_blocks = tl.cdiv(y_size, y_block_size)

        grad_key_ptr += pid * head_stride
        grad_value_ptr += pid * head_stride
        key_ptr += pid * head_stride
        value_ptr += pid * head_stride

        score_scale = softmax_scale * language.log2e
        n_strides = tl.arange(0, y_block_size) * y_stride
        x_strides = tl.arange(0, x_block_size) * x_stride

        for n_block in range(0, num_y_blocks):
            if is_causal:
                n_offset = n_block * y_block_size
            else:
                n_offset = 0

            grad_query_block_ptr = tl.make_block_ptr(
                grad_query_ptr + pid * head_stride,
                shape=(y_size, x_size),
                strides=(y_stride, x_stride),
                offsets=(n_offset, 0),
                block_shape=(y_block_size, x_block_size),
                order=(1, 0),
            )
            grad_output_block_ptr = tl.make_block_ptr(
                grad_output_ptr + pid * head_stride,
                shape=(y_size, x_size),
                strides=(y_stride, x_stride),
                offsets=(n_offset, 0),
                block_shape=(y_block_size, x_block_size),
                order=(1, 0),
            )
            query_block_ptr = tl.make_block_ptr(
                query_ptr + pid * head_stride,
                shape=(y_size, x_size),
                strides=(y_stride, x_stride),
                offsets=(n_offset, 0),
                block_shape=(y_block_size, x_block_size),
                order=(1, 0),
            )
            output_block_ptr = tl.make_block_ptr(
                output_ptr + pid * head_stride,
                shape=(y_size, x_size),
                strides=(y_stride, x_stride),
                offsets=(n_offset, 0),
                block_shape=(y_block_size, x_block_size),
                order=(1, 0),
            )
            log2sum_block_ptr = tl.make_block_ptr(
                log2sum_ptr + pid * y_size,
                shape=(y_size,),
                strides=(1,),
                offsets=(n_offset,),
                block_shape=(y_block_size,),
                order=(0,),
            )
            delta_block_ptr = tl.make_block_ptr(
                delta_ptr + pid * y_size,
                shape=(y_size,),
                strides=(1,),
                offsets=(n_offset,),
                block_shape=(y_block_size,),
                order=(0,),
            )

            if mask_ptr is not None:
                grad_mask_block_ptr = tl.make_block_ptr(
                    grad_mask_ptr + pid * mask_head_stride,
                    shape=(y_size, y_size),
                    strides=(mask_y_stride, mask_x_stride),
                    offsets=(0, n_block * y_block_size),
                    block_shape=(y_block_size, y_block_size),
                    order=(1, 0),
                )
                mask_block_ptr = tl.make_block_ptr(
                    mask_ptr + pid * mask_head_stride,
                    shape=(y_size, y_size),
                    strides=(mask_y_stride, mask_x_stride),
                    offsets=(0, n_block * y_block_size),
                    block_shape=(y_block_size, y_block_size),
                    order=(1, 0),
                )

            grad_value = tl.zeros((y_block_size, x_block_size), dtype)
            grad_key = tl.zeros((y_block_size, x_block_size), dtype)
            ptr_offsets = n_strides[:, None] + x_strides[None, :]
            key = tl.load(key_ptr + ptr_offsets)
            value = tl.load(value_ptr + ptr_offsets)
            n_offsets = n_offset + tl.arange(0, y_block_size)

            for m_offset in range(n_offset, y_size, y_block_size):
                query = tl.load(query_block_ptr)
                m_offsets = tl.arange(0, y_block_size) + m_offset

                if is_causal:
                    condition = m_offsets[:, None] >= n_offsets[None, :]
                    score = tl.where(condition, 0.0, float("-inf"))
                elif mask_ptr is not None:
                    mask = tl.load(mask_block_ptr)
                    mask *= language.log2e
                    score = mask
                else:
                    score = tl.zeros((y_block_size, y_block_size), dtype)

                score += language.dot(query, tl.trans(key), use_accelerator, dtype) * score_scale
                log2sum = tl.load(log2sum_block_ptr)
                alpha = tl.math.exp2(score - log2sum[:, None]).to(dtype)
                grad_output = tl.load(grad_output_block_ptr)

                if dropout_p > language.eps:
                    output = tl.load(output_block_ptr)
                    dropout_scale = 1.0 - dropout_p + language.eps
                    grad_dropout = tl.where(output > 0.0, dropout_scale, 0.0).to(dtype)
                    grad_output *= grad_dropout

                grad_value += language.dot(tl.trans(alpha), grad_output, use_accelerator, dtype)
                delta = tl.load(delta_block_ptr)
                grad_alpha = tl.zeros((y_block_size, y_block_size), dtype) - delta[:, None]
                grad_alpha += language.dot(grad_output, tl.trans(value), use_accelerator, dtype)
                grad_softmax = (alpha * grad_alpha * softmax_scale).to(dtype)
                grad_key += language.dot(tl.trans(grad_softmax), query, use_accelerator, dtype)
                grad_query = tl.load(grad_query_block_ptr)
                grad_query += language.dot(grad_softmax, key, use_accelerator, dtype)
                tl.store(grad_query_block_ptr, grad_query)

                grad_query_block_ptr = tl.advance(grad_query_block_ptr, (y_block_size, 0))
                grad_output_block_ptr = tl.advance(grad_output_block_ptr, (y_block_size, 0))
                query_block_ptr = tl.advance(query_block_ptr, (y_block_size, 0))
                output_block_ptr = tl.advance(output_block_ptr, (y_block_size, 0))
                delta_block_ptr = tl.advance(delta_block_ptr, (y_block_size,))
                log2sum_block_ptr = tl.advance(log2sum_block_ptr, (y_block_size,))

                if mask_ptr is not None:
                    tl.store(grad_mask_block_ptr, (grad_softmax / softmax_scale).to(dtype))
                    mask_block_ptr = tl.advance(mask_block_ptr, (y_block_size, 0))
                    grad_mask_block_ptr = tl.advance(grad_mask_block_ptr, (y_block_size, 0))

            tl.store(grad_key_ptr + ptr_offsets, grad_key)
            tl.store(grad_value_ptr + ptr_offsets, grad_value)
            n_strides += y_block_size * y_stride
