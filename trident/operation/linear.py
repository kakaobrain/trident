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
        return Linear.__forward(*args, **kwargs)

    @staticmethod
    def setup_context(ctx, inputs, output):
        inp, wgt, bis, act = inputs
        ctx.save_for_backward(inp, wgt, output)
        ctx.act = act

    @staticmethod
    def backward(ctx, *grad_outputs):
        return Linear.__backward(*grad_outputs, *ctx.saved_tensors, ctx.act)

    @staticmethod
    def __forward(inp, wgt, bis, act):
        assert inp.is_cuda and inp.is_contiguous()
        assert wgt.is_cuda and wgt.is_contiguous()
        assert inp.shape[1] == wgt.shape[1]

        if bis is not None:
            assert bis.is_cuda and bis.is_contiguous()
            assert wgt.shape[0] == bis.shape[0] if bis.dim() == 1 else bis.shape[1]
            bis_st = bis.stride(0) if bis.dim() == 1 else bis.stride(1)
        else:
            bis_st = 0

        m, k = inp.shape
        n, _ = wgt.shape
        grid = lambda meta: (
            triton.cdiv(m, meta["blk_sz_m"]),
            triton.cdiv(n, meta["blk_sz_n"]),
        )
        y = torch.empty((m, n), device="cuda")

        kernel.Linear.forward[grid](
            inp,
            inp.stride(0),
            inp.stride(1),
            y,
            y.stride(0),
            y.stride(1),
            wgt,
            wgt.stride(0),
            wgt.stride(1),
            bis,
            bis_st,
            m,
            k,
            n,
            act=act,
        )

        return y

    @staticmethod
    def __backward(grad_out, inp, wgt, out, act):
        return kernel.Linear.backward(grad_out, inp, wgt, out, act)
