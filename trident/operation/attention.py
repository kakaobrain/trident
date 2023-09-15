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
        query, key, value, dropout_p, is_causal, softmax_scale, use_accelerator = args

        util.push_trace("Attention.__forward")
        output, log_sum_exp = Attention.__forward(
            query, key, value, dropout_p, is_causal, softmax_scale, use_accelerator
        )
        util.pop_trace()

        ctx.save_for_backward(query, key, value, output, log_sum_exp)
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.softmax_scale = softmax_scale
        ctx.use_accelerator = use_accelerator

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        (grad_output,) = grad_outputs
        query, key, value, output, log_sum_exp = ctx.saved_tensors

        util.push_trace("Attention.__backward")
        grad_query, grad_key, grad_value = Attention.__backward(
            grad_output,
            query,
            key,
            value,
            output,
            log_sum_exp,
            ctx.softmax_scale,
            ctx.dropout_p,
            ctx.is_causal,
            ctx.use_accelerator,
        )
        util.pop_trace()

        return grad_query, grad_key, grad_value, None, None, None, None

    @staticmethod
    def __forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dropout_p: torch.float32,
        is_causal: torch.bool,
        softmax_scale: torch.float32,
        use_accelerator: torch.bool,
    ):
        assert query.shape[-1] == key.shape[-1] and key.shape[-1] == value.shape[-1]
        assert key.shape[-1] in {16, 32, 64, 128}

        factory_kwargs = {"device": query.device, "dtype": query.dtype}
        num_batches, num_heads, y_size, x_size = query.shape
        output = torch.empty_like(query)
        log2sum = torch.empty(num_batches, num_heads, y_size, **factory_kwargs)

        def grid(meta):
            num_m_blocks = triton.cdiv(y_size, meta["y_block_size"])
            return (num_batches * num_heads * num_m_blocks,)

        util.push_trace("kernel.Attention.forward")
        kernel.Attention.forward[grid](
            output,
            log2sum,
            query,
            key,
            value,
            y_size,
            x_size,
            query.stride(1),
            query.stride(2),
            query.stride(3),
            dropout_p,
            torch.random.seed(),
            is_causal,
            softmax_scale,
            use_accelerator,
            util.dtype(output.dtype),
            x_block_size=triton.next_power_of_2(x_size),
        )
        util.pop_trace()

        return output, log2sum

    @staticmethod
    def __backward(
        grad_output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        log2sum: torch.Tensor,
        softmax_scale: torch.float32,
        dropout_p: torch.float32,
        is_causal: torch.bool,
        use_accelerator: torch.bool,
    ):
        num_batches, num_heads, y_size, x_size = output.shape
        grad_query = torch.zeros_like(query)
        grad_key = torch.empty_like(key)
        grad_value = torch.empty_like(value)
        delta = torch.empty_like(log2sum)

        def grid(meta):
            return (num_batches * num_heads * y_size,)

        util.push_trace("kernel.Softmax.backward_delta")
        kernel.Softmax.backward_delta[grid](
            delta, output, grad_output, num_batches * num_heads * y_size, x_size, x_size, 1, util.dtype(delta.dtype)
        )
        util.pop_trace()

        def grid(meta):
            return (num_batches * num_heads,)

        util.push_trace("kernel.Attention.backward")
        kernel.Attention.backward[grid](
            grad_query,
            grad_key,
            grad_value,
            grad_output,
            query,
            key,
            value,
            y_size,
            x_size,
            query.stride(1),
            query.stride(2),
            query.stride(3),
            output,
            log2sum,
            delta,
            dropout_p,
            is_causal,
            softmax_scale,
            use_accelerator,
            util.dtype(grad_query.dtype),
            64,
            triton.next_power_of_2(x_size),
            num_warps=4 if x_size <= 64 else 8,
        )
        util.pop_trace()

        return grad_query, grad_key, grad_value
