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


class PReLU(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        x, negative_slopes = args

        assert x.is_cuda and x.is_contiguous()
        assert negative_slopes.is_cuda and negative_slopes.is_contiguous()

        size_0, size_1 = x.shape
        y = torch.empty_like(x)

        assert y.is_cuda and y.is_contiguous()

        block_size = min(triton.next_power_of_2(size_1), 1 << 14)

        grid = lambda meta: (size_0, triton.cdiv(size_1, meta['block_size']),)
        kernel.PReLU.forward[grid](x, x.stride(0),
                                   y, y.stride(0),
                                   negative_slopes, size_1,
                                   block_size=block_size)

        return y

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, negative_slopes, = inputs
        ctx.save_for_backward(x)
        ctx.negative_slopes = negative_slopes

    @staticmethod
    def backward(ctx, *grad_outputs):
        x, = ctx.saved_tensors
        w = ctx.negative_slopes

        assert x.is_cuda and x.is_contiguous()
        assert w.is_cuda and w.is_contiguous()

        x_shape_0, x_shape_1 = x.shape
        dx = torch.empty_like(x)
        dw = torch.empty_like(x)

        def get_block_size():
            return min(triton.next_power_of_2(x_shape_1), 1 << 14)

        grid = lambda meta: (x_shape_0, triton.cdiv(x_shape_1, meta['block_size']),)

        kernel.PReLU.backward[grid](x, x.stride(0),
                                    dx, dx.stride(0),
                                    w, dw, x_shape_1,
                                    block_size=get_block_size())

        return grad_outputs[0] * dx, grad_outputs[0] * dw, None
