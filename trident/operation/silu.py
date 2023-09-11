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


class SiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        (input,) = args

        util.push_trace("SiLU.__forward")
        output = SiLU.__forward(input)
        util.pop_trace()

        ctx.save_for_backward(input)

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        (grad_output,) = grad_outputs
        (input,) = ctx.saved_tensors

        util.push_trace("SiLU.__backward")
        grad_input = SiLU.__backward(grad_output, input)
        util.pop_trace()

        return grad_input, None

    @staticmethod
    def __forward(input: torch.Tensor):
        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        x_size = input.numel()
        output = torch.empty(x_size, **factory_kwargs)

        def grid(meta):
            return (triton.cdiv(x_size, meta["x_block_size"]),)

        util.push_trace("kernel.SiLU.forward")
        kernel.SiLU.forward[grid](output, input, x_size, util.dtype(output.dtype))
        util.pop_trace()

        return output

    @staticmethod
    def __backward(grad_output: torch.Tensor, input: torch.Tensor):
        x_size = input.numel()
        grad_input = torch.empty_like(input)

        def grid(meta):
            return (triton.cdiv(x_size, meta["x_block_size"]),)

        util.push_trace("kernel.SiLU.backward")
        kernel.SiLU.backward[grid](grad_input, grad_output, input, x_size, util.dtype(grad_input.dtype))
        util.pop_trace()

        return grad_input
