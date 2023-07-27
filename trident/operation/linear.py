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
        return Linear.__forward(*args, **kwargs)

    @staticmethod
    def setup_context(ctx, inputs, output):
        inp, wgt, bis, act = inputs
        ctx.save_for_backward(inp, wgt, bis, output)
        ctx.act = act

    @staticmethod
    def backward(ctx, *grad_outputs):
        return Linear.__backward(*grad_outputs, *ctx.saved_tensors, ctx.act)

    @staticmethod
    def __forward(inp, wgt, bis, act):
        assert inp.is_contiguous()
        assert wgt.is_contiguous()
        assert inp.shape[1] == wgt.shape[1]

        if bis is not None:
            assert bis.is_contiguous()
            assert wgt.shape[0] == bis.shape[0] if bis.dim() == 1 else bis.shape[1]
            bis_st = bis.stride(0) if bis.dim() == 1 else bis.stride(1)
        else:
            bis_st = 0

        ctor_args = {"device": inp.device, "dtype": inp.dtype}

        m, k = inp.shape
        n, _ = wgt.shape
        grid = lambda meta: (
            triton.cdiv(m, meta["blk_sz_m"]),
            triton.cdiv(n, meta["blk_sz_n"]),
        )
        y = torch.empty((m, n), **ctor_args)

        kernel.Linear.forward[grid](
            inp, y, wgt, bis, bis_st, m, k, n, act=act, dtype=util.dtype(inp.dtype)
        )

        return y

    @staticmethod
    def __backward(grad_out, inp, wgt, bis, out, act):
        m, k = inp.shape
        n, _ = wgt.shape

        ctor_args = {"device": inp.device, "dtype": inp.dtype}

        grad_act = torch.empty_like(grad_out)
        grad_bis = torch.zeros(n, **ctor_args) if bis is not None else None

        def grid(meta):
            return [n]

        kernel.Linear.backward_bias[grid](
            grad_out,
            out,
            grad_act,
            grad_bis,
            m,
            n,
            act=act,
            blk_sz=util.block_size(n, grad_act.element_size()),
            num_warps=util.num_warps(n, grad_act.element_size()),
        )

        def grid(meta):
            return [
                triton.cdiv(max(m, n), meta["blk_sz_m"]),
                triton.cdiv(k, meta["blk_sz_k"]),
            ]

        grad_inp = torch.empty_like(inp)
        grad_wgt = torch.empty_like(wgt)

        kernel.Linear.backward[grid](
            grad_act,
            wgt,
            inp,
            grad_inp,
            grad_wgt,
            m,
            n,
            k,
            dtype=util.dtype(inp.dtype),
        )

        return grad_inp, grad_wgt, grad_bis, None
