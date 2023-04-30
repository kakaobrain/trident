"""
Copyright 2023 ⓒ Kakao Brain Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import triton
from trident import kernel
from trident import function


class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim=None):
        """
        Applies a softmax function to the input tensor and return the result.

        :param ctx: It can be used to store arbitrary data that can be then retrieved during the backward pass.
        :param x: Input tensor.
        :param dim: A dimension along which softmax will be computed.
        :return: Output tensor.
        """
        assert dim == 1

        y = function.softmax(x, dim)
        ctx.save_for_backward(y)

        return y

    @staticmethod
    def backward(ctx, t):
        assert t.is_cuda and t.is_contiguous()

        y = ctx.saved_tensors[0]

        assert y.is_cuda and y.is_contiguous()

        m, n = y.shape
        block_size_n = triton.next_power_of_2(n)

        def get_num_warps():
            if block_size_n >= 8192:
                return 16
            if block_size_n >= 2048:
                return 4
            return 2

        d = torch.empty_like(y)

        assert d.is_cuda and d.is_contiguous

        kernel.softmax_backward[(m, )](y, y.stride(0), y.stride(1),
                                       t, t.stride(0), t.stride(1),
                                       d, d.stride(0), d.stride(1),
                                       n,
                                       BLOCK_SIZE_N=block_size_n, num_warps=get_num_warps())
        return d, None