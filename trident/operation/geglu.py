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

import torch
import triton

from trident import kernel, util


class GEGLU(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        input, weight, bias, use_accelerator = args
        return GEGLU.__forward(input, weight, bias, use_accelerator)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias, use_accelerator = inputs
        _, state_gate = output
        ctx.save_for_backward(input, weight, bias, state_gate)
        ctx.use_accelerator = False

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, _ = grad_outputs
        input, weight, bias, state_gate = ctx.saved_tensors
        return GEGLU.__backward(grad_output, input, weight, bias, state_gate, ctx.use_accelerator)

    @staticmethod
    def __forward(input, weight, bias, use_accelerator):
        if input.dim() == 2:
            num_batches = 1
            m_size, k_size = input.shape
        else:
            num_batches, m_size, k_size = input.shape

        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        n_size, _ = weight.shape
        x_size = n_size // 2
        output = torch.empty(num_batches, m_size, x_size, **factory_kwargs)
        state_gate = torch.empty(num_batches, m_size, n_size, **factory_kwargs)

        def grid(meta):
            num_m_blocks = triton.cdiv(m_size, meta["m_block_size"])
            num_x_blocks = triton.cdiv(x_size, meta["x_block_size"])
            return (num_batches * num_m_blocks * num_x_blocks,)

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
            use_accelerator,
            util.dtype(input.dtype),
        )

        return output, state_gate

    @staticmethod
    def __backward(grad_output, input, weight, bias, state_gate, use_accelerator):
        if input.dim() == 2:
            num_batches = 1
            m_size, k_size = input.shape
        else:
            num_batches, m_size, k_size = input.shape

        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        n_size, _ = weight.shape
        x_size = n_size // 2
        grad_state_gate = torch.empty_like(state_gate)
        grad_input = torch.empty_like(input)
        grad_weight_staging = torch.empty(num_batches, n_size, k_size, **factory_kwargs)

        def grid(meta):
            return (num_batches * m_size,)

        kernel.GEGLU.backward[grid](
            grad_state_gate, grad_output, state_gate, m_size, n_size, x_size, triton.next_power_of_2(x_size)
        )

        def grid(meta):
            num_m_blocks = triton.cdiv(m_size, meta["m_block_size"])
            num_k_blocks = triton.cdiv(k_size, meta["k_block_size"])
            return (num_batches * num_m_blocks * num_k_blocks,)

        kernel.Linear.backward[grid](
            grad_input,
            grad_state_gate,
            weight,
            m_size,
            n_size,
            k_size,
            use_accelerator,
            util.dtype(grad_input.dtype),
        )

        def grid(meta):
            num_n_blocks = triton.cdiv(n_size, meta["n_block_size"])
            num_k_blocks = triton.cdiv(k_size, meta["k_block_size"])
            return (num_batches * num_n_blocks * num_k_blocks,)

        kernel.Linear.backward_weight[grid](
            grad_weight_staging,
            grad_state_gate,
            input,
            m_size,
            n_size,
            k_size,
            use_accelerator,
            util.dtype(grad_weight_staging.dtype),
        )

        grad_weight = torch.sum(grad_weight_staging, 0)

        if bias is not None:
            grad_bias_staging = torch.empty(num_batches, n_size, **factory_kwargs)

            def grid(meta):
                return (num_batches * n_size,)

            kernel.Linear.backward_bias[grid](
                grad_bias_staging,
                grad_state_gate,
                m_size,
                n_size,
                util.dtype(grad_bias_staging.dtype),
            )

            grad_bias = torch.sum(grad_bias_staging, 0)
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None
