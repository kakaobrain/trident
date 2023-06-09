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


class Linear(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        x, w, b, activation = args

        assert x.is_cuda and x.is_contiguous()
        assert w.is_cuda and w.is_contiguous()
        assert x.shape[1] == w.shape[1]

        if b is not None:
            assert b.is_cuda and b.is_contiguous()
            assert w.shape[0] == b.shape[0] if b.dim() == 1 else b.shape[1]

        m, k = x.shape
        n, _ = w.shape
        grid = lambda meta: (triton.cdiv(m, meta['BLOCK_SIZE_M']), triton.cdiv(n, meta['BLOCK_SIZE_N']),)
        y = torch.empty((m, n), device='cuda')

        if b is None:
            kernel.linear[grid](x, x.stride(0), x.stride(1),
                                y, y.stride(0), y.stride(1),
                                w, w.stride(0), w.stride(1),
                                None, 0,
                                m, k, n,
                                ACTIVATION=activation,
                                BLOCK_SIZE_K=16)
        else:
            assert b.is_cuda and b.is_contiguous()
            assert w.shape[0] == b.shape[0] if b.dim() == 1 else b.shape[1]

            kernel.linear[grid](x, x.stride(0), x.stride(1),
                                y, y.stride(0), y.stride(1),
                                w, w.stride(0), w.stride(1),
                                b, b.stride(0) if b.dim() == 1 else b.stride(1),
                                m, k, n,
                                ACTIVATION=activation,
                                BLOCK_SIZE_K=16)

        return y

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("The backward of Linear isn't implemented.")
