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

from typing import Any, Tuple

import torch
import triton

from trident import kernel, util


class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        query, key, value, is_causal, softmax_scale, use_accelerator = args

        util.push_trace("Attention.__forward")
        output, log_sum_exp, grid = Attention.__forward(query, key, value, is_causal, softmax_scale, use_accelerator)
        util.pop_trace()

        ctx.save_for_backward(query, key, value, output, log_sum_exp)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.embedding_size = key.shape[-1]
        ctx.is_causal = is_causal
        ctx.use_accelerator = use_accelerator

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        (grad_output,) = grad_outputs

        query, key, value, output, log_sum_exp = ctx.saved_tensors
        grid = ctx.grid
        softmax_scale = ctx.softmax_scale
        embedding_size = ctx.embedding_size
        is_causal = ctx.is_causal
        use_accelerator = ctx.use_accelerator

        util.push_trace("Attention.__backward")
        grad_query, grad_key, grad_value = Attention.__backward(
            grad_output,
            query,
            key,
            value,
            output,
            log_sum_exp,
            grid,
            softmax_scale,
            embedding_size,
            is_causal,
            use_accelerator,
        )
        util.pop_trace()

        return grad_query, grad_key, grad_value, None, None, None

    @staticmethod
    def __forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool,
        softmax_scale: float,
        use_accelerator: bool,
    ):
        assert query.shape[-1] == key.shape[-1] and key.shape[-1] == value.shape[-1]
        assert key.shape[-1] in {16, 32, 64, 128}

        output = torch.empty_like(query)
        m_block_size = 64
        n_block_size = 64
        grid = (triton.cdiv(query.shape[2], m_block_size), query.shape[0] * query.shape[1], 1)
        log_sum_exp = torch.empty(
            (query.shape[0] * query.shape[1], query.shape[2]), device=query.device, dtype=torch.float32
        )
        num_warps = 4 if key.shape[-1] <= 64 else 8

        util.push_trace("kernel.Attention.forward")
        kernel.Attention.forward[grid](
            output,
            log_sum_exp,
            query,
            key,
            value,
            softmax_scale,
            query.stride(1),
            query.stride(2),
            query.stride(3),
            key.stride(2),
            key.stride(3),
            value.stride(2),
            value.stride(3),
            output.stride(2),
            output.stride(3),
            query.shape[2],
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            embedding_size=key.shape[-1],
            is_causal=is_causal,
            use_accelerator=use_accelerator,
            dtype=util.dtype(query.dtype),
            num_warps=num_warps,
        )
        util.pop_trace()

        return output, log_sum_exp, grid

    @staticmethod
    def __backward(
        grad_output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        log_sum_exp: torch.Tensor,
        grid: Tuple,
        softmax_scale: float,
        embedding_size,
        is_causal: bool,
        use_accelerator: bool,
    ):
        block_size = 64
        grad_output = grad_output.contiguous()
        grad_query = torch.zeros_like(query)
        grad_key = torch.empty_like(key)
        grad_value = torch.empty_like(value)
        delta = torch.empty_like(log_sum_exp)

        num_batches, num_heads, y_size, x_size = output.shape

        util.push_trace("kernel.Softmax.backward_delta")
        kernel.Softmax.backward_delta[(num_batches * num_heads * y_size,)](
            delta, output, grad_output, num_batches * num_heads * y_size, x_size, x_size, 1, util.dtype(delta.dtype)
        )
        util.pop_trace()

        util.push_trace("kernel.Attention.backward")
        kernel.Attention.backward[(grid[1],)](
            grad_query,
            grad_key,
            grad_value,
            query,
            key,
            value,
            grad_output,
            log_sum_exp,
            delta,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),
            key.stride(2),
            key.stride(3),
            query.shape[1],
            query.shape[2],
            grid[0],
            softmax_scale,
            m_block_size=block_size,
            n_block_size=block_size,
            embedding_size=embedding_size,
            is_causal=is_causal,
            use_accelerator=use_accelerator,
            dtype=util.dtype(query.dtype),
            num_warps=8,
        )
        util.pop_trace()

        return grad_query, grad_key, grad_value
