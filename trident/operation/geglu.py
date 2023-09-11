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

from typing import Any

import torch
import triton

from trident import kernel, util


class GEGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, weight, bias, use_accelerator = args

        util.push_trace("GEGLU.__forward")
        output, state_gate = GEGLU.__forward(input, weight, bias, use_accelerator)
        util.pop_trace()

        ctx.save_for_backward(input, weight, bias, state_gate)
        ctx.use_accelerator = False

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        (grad_output,) = grad_outputs
        input, weight, bias, state_gate = ctx.saved_tensors

        util.push_trace("GEGLU.__backward")
        grad_input, grad_weight, grad_bias = GEGLU.__backward(
            grad_output, input, weight, bias, state_gate, ctx.use_accelerator
        )
        util.pop_trace()

        return grad_input, grad_weight, grad_bias, None

    @staticmethod
    def __forward(input, weight, bias, use_accelerator):
        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        num_batches, m_size, k_size = input.shape
        n_size, _ = weight.shape
        x_size = n_size // 2
        output = torch.empty(num_batches, m_size, x_size, **factory_kwargs)
        state_gate = torch.empty(num_batches, m_size, n_size, **factory_kwargs)

        def grid(meta):
            num_m_blocks = triton.cdiv(m_size, meta["m_block_size"])
            num_x_blocks = triton.cdiv(x_size, meta["x_block_size"])
            return (num_batches * num_m_blocks * num_x_blocks,)

        util.push_trace("kernel.GEGLU.forward")
        kernel.GEGLU.forward[grid](
            output,
            state_gate,
            input,
            weight,
            bias,
            m_size,
            n_size,
            k_size,
            x_size,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            weight.stride(0),
            weight.stride(1),
            use_accelerator,
            util.dtype(input.dtype),
        )
        util.pop_trace()

        return output, state_gate

    @staticmethod
    def __backward(grad_output, input, weight, bias, state_gate, use_accelerator):
        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        num_batches, m_size, k_size = input.shape
        n_size, _ = weight.shape
        x_size = n_size // 2
        grad_state_gate = torch.empty_like(state_gate)
        grad_input = torch.empty_like(input)
        grad_weight_staging = torch.empty(num_batches, n_size, k_size, **factory_kwargs)

        def grid(meta):
            return (num_batches * m_size,)

        util.push_trace("kernel.GEGLU.backward")
        kernel.GEGLU.backward[grid](
            grad_state_gate, grad_output, state_gate, m_size, n_size, x_size, triton.next_power_of_2(x_size)
        )
        util.pop_trace()

        def grid(meta):
            num_m_blocks = triton.cdiv(m_size, meta["m_block_size"])
            num_k_blocks = triton.cdiv(k_size, meta["k_block_size"])
            return (num_batches * num_m_blocks * num_k_blocks,)

        util.push_trace("kernel.Linear.backward")
        kernel.Linear.backward[grid](
            grad_input,
            grad_state_gate,
            weight,
            m_size,
            n_size,
            k_size,
            input.stride(1),
            input.stride(2),
            weight.stride(0),
            weight.stride(1),
            use_accelerator,
            util.dtype(grad_input.dtype),
        )
        util.pop_trace()

        def grid(meta):
            num_n_blocks = triton.cdiv(n_size, meta["n_block_size"])
            num_k_blocks = triton.cdiv(k_size, meta["k_block_size"])
            return (num_batches * num_n_blocks * num_k_blocks,)

        util.push_trace("kernel.Linear.backward_weight")
        kernel.Linear.backward_weight[grid](
            grad_weight_staging,
            grad_state_gate,
            input,
            m_size,
            n_size,
            k_size,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            use_accelerator,
            util.dtype(grad_weight_staging.dtype),
        )
        util.pop_trace()

        util.push_trace("torch.sum")
        grad_weight = torch.sum(grad_weight_staging, 0)
        util.pop_trace()

        if bias is not None:
            grad_bias_staging = torch.empty(num_batches, n_size, **factory_kwargs)

            def grid(meta):
                return (num_batches * n_size,)

            util.push_trace("kernel.Linear.backward_bias")
            kernel.Linear.backward_bias[grid](
                grad_bias_staging,
                grad_state_gate,
                m_size,
                n_size,
                util.dtype(grad_bias_staging.dtype),
            )
            util.pop_trace()

            util.push_trace("torch.sum")
            grad_bias = torch.sum(grad_bias_staging, 0)
            util.pop_trace()
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias
