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


class Linear(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        input, weight, bias, use_accelerator = args
        return Linear.__forward(input, weight, bias, use_accelerator)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias, use_accelerator = inputs
        ctx.save_for_backward(input, weight, bias, output)
        ctx.use_accelerator = use_accelerator

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_output,) = grad_outputs
        input, weight, bias, output = ctx.saved_tensors
        return Linear.__backward(grad_output, output, input, weight, bias, ctx.use_accelerator)

    @staticmethod
    def __forward(input, weight, bias, use_accelerator):
        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        num_batches, m_size, k_size = input.shape
        n_size, _ = weight.shape
        output = torch.empty(num_batches, m_size, n_size, **factory_kwargs)

        def grid(meta):
            num_m_blocks = triton.cdiv(m_size, meta["m_block_size"])
            num_n_blocks = triton.cdiv(n_size, meta["n_block_size"])
            return (num_batches * num_m_blocks * num_n_blocks,)

        kernel.Linear.forward[grid](
            output,
            input,
            weight,
            bias,
            m_size,
            n_size,
            k_size,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            weight.stride(0),
            weight.stride(1),
            use_accelerator,
            util.dtype(input.dtype),
        )

        return output

    @staticmethod
    def __backward(grad_output, output, input, weight, bias, use_accelerator):
        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        num_batches, m_size, k_size = input.shape
        n_size, _ = weight.shape
        grad_input = torch.empty_like(input)
        grad_weight_staging = torch.empty(num_batches, n_size, k_size, **factory_kwargs)

        def grid(meta):
            num_m_blocks = triton.cdiv(m_size, meta["m_block_size"])
            num_k_blocks = triton.cdiv(k_size, meta["k_block_size"])
            return (num_batches * num_m_blocks * num_k_blocks,)

        kernel.Linear.backward[grid](
            grad_input,
            grad_output,
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

        def grid(meta):
            num_n_blocks = triton.cdiv(n_size, meta["n_block_size"])
            num_k_blocks = triton.cdiv(k_size, meta["k_block_size"])
            return (num_batches * num_n_blocks * num_k_blocks,)

        kernel.Linear.backward_weight[grid](
            grad_weight_staging,
            grad_output,
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

        grad_weight = torch.sum(grad_weight_staging, 0)

        if bias is not None:
            grad_bias_staging = torch.empty(num_batches, n_size, **factory_kwargs)

            def grid(meta):
                return (num_batches * n_size,)

            kernel.Linear.backward_bias[grid](
                grad_bias_staging,
                grad_output,
                m_size,
                n_size,
                util.dtype(grad_bias_staging.dtype),
            )

            grad_bias = torch.sum(grad_bias_staging, 0)
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None
