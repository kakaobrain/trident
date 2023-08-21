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
        _, linear = output
        ctx.save_for_backward(input, weight, bias, linear)
        ctx.use_accelerator = False

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, _ = grad_outputs
        linear, input, weight, bias = ctx.saved_tensors
        return GEGLU.__backward(grad_output, linear, input, weight, bias, ctx.use_accelerator)

    @staticmethod
    def __forward(input, weight, bias, use_accelerator):
        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        m_size, k_size = input.shape
        n_size, _ = weight.shape
        output = torch.empty((m_size, n_size), **factory_kwargs)
        linear = torch.empty((m_size, n_size), **factory_kwargs)

        def grid(meta):
            return (triton.cdiv(m_size, meta["m_block_size"]) * triton.cdiv(n_size, meta["n_block_size"]),)

        kernel.GEGLU.forward[grid](
            output,
            linear,
            input,
            weight,
            bias,
            m_size,
            n_size,
            k_size,
            use_accelerator,
            dtype=util.dtype(input.dtype),
        )

        return output, linear

    @staticmethod
    def __backward(grad_output, input, weight, bias, linear, use_accelerator):
        m_size, k_size = input.shape
        n_size, _ = weight.shape
        grad_input = torch.empty_like(input)
        grad_weight = torch.empty_like(weight)

        def grid(meta):
            return (triton.cdiv(m_size, meta["m_block_size"]) * triton.cdiv(k_size, meta["k_block_size"]),)

        kernel.GEGLU.backward[grid](
            grad_input,
            grad_output,
            weight,
            linear,
            m_size,
            n_size,
            k_size,
            use_accelerator,
            dtype=util.dtype(input.dtype),
        )

        def grid(meta):
            return (triton.cdiv(n_size, meta["n_block_size"]) * triton.cdiv(k_size, meta["k_block_size"]),)

        kernel.GEGLU.backward_weight[grid](
            grad_weight,
            grad_output,
            input,
            linear,
            m_size,
            n_size,
            k_size,
            use_accelerator,
            dtype=util.dtype(input.dtype),
        )

        if bias is not None:
            grad_bias = torch.empty_like(bias)

            def grid(meta):
                return (n_size,)

            kernel.GEGLU.backward_bias[grid](
                grad_bias,
                grad_output,
                linear,
                m_size,
                n_size,
                dtype=util.dtype(input.dtype),
            )
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None
