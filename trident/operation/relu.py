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

from trident import kernel


class ReLU(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        x, = args

        assert x.is_cuda and x.is_contiguous()

        block_size = min(triton.next_power_of_2(x.shape[1]), 1 << 14)
        y = torch.empty_like(x)

        assert y.is_cuda and y.is_contiguous()

        grid = lambda meta: (x.shape[0], triton.cdiv(x.shape[1], meta['block_size']))
        kernel.ReLU.forward[grid](x, y, x.stride(0), x.shape[1],
                                  block_size=block_size)

        return y

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, = inputs

        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, = ctx.saved_tensors

        assert x.is_cuda and x.is_contiguous()

        block_size = min(triton.next_power_of_2(x.shape[1]), 1 << 14)
        dx = torch.empty_like(x)

        assert dx.is_cuda and dx.is_contiguous()

        grid = lambda meta: (x.shape[0], triton.cdiv(x.shape[1], meta['block_size']))
        kernel.ReLU.backward[grid](dx, x, x.stride(0), x.shape[1],
                                   block_size=block_size)

        return grad_outputs[0] * dx, None
