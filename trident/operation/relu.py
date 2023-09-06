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


class ReLU(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        (input,) = args
        return ReLU.__forward(input)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (input,) = inputs
        ctx.save_for_backward(input)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        (input,) = ctx.saved_tensors
        return ReLU.__backward(grad_output, input)

    @staticmethod
    def __forward(input: torch.Tensor):
        x_size = input.numel()
        output = torch.empty_like(input)

        def grid(meta):
            return (triton.cdiv(x_size, meta["x_block_size"]),)

        kernel.ReLU.forward[grid](output, input, x_size, util.dtype(output.dtype))

        return output

    @staticmethod
    def __backward(grad_output: torch.Tensor, input: torch.Tensor):
        x_size = input.numel()
        grad_input = torch.empty_like(input)

        def grid(meta):
            return [triton.cdiv(x_size, meta["x_block_size"])]

        kernel.ReLU.backward[grid](grad_input, grad_output, input, x_size, util.dtype(grad_input.dtype))

        return grad_input
