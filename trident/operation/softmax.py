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


class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        x, dim = args

        assert x.is_cuda and x.is_contiguous()
        assert dim == 1

        m, n = x.shape
        block_size_n = triton.next_power_of_2(n)

        def get_num_stages():
            if block_size_n <= 2048:
                return 4
            if block_size_n <= 4096:
                return 3
            return 2

        def get_num_warps():
            if block_size_n >= 8192:
                return 16
            if block_size_n >= 4096:
                return 8
            if block_size_n >= 2048:
                return 4
            return 2

        y = torch.empty_like(x)

        assert y.is_cuda and x.is_contiguous

        kernel.softmax_forward[(m,)](x, x.stride(0), x.stride(1), y, y.stride(0), y.stride(1), m, n,
                                     BLOCK_SIZE_N=block_size_n, num_stages=get_num_stages(), num_warps=get_num_warps())

        return y

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, *grad_outputs):
        t, = grad_outputs

        assert t.is_cuda and t.is_contiguous()

        y = ctx.saved_tensors[0]

        assert y.is_cuda and y.is_contiguous()

        grad_x = torch.empty_like(y)

        assert grad_x.is_cuda and grad_x.is_contiguous

        for i, s in enumerate(y):
            jacob = torch.diagflat(s) - torch.mm(s.unsqueeze(0).t(), s.unsqueeze(0))
            grad_x[i] = torch.mm(t[i].unsqueeze(0), jacob)

        return grad_x, None
