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

import triton

from trident import language


class BatchNorm:
    @staticmethod
    @triton.jit
    def forward(
        inp_ptr,
        vec_sz,
        bt_sz,
        wgt_ptr,
        bis_ptr,
        eps,
        running_mean_ptr,
        running_var_ptr,
        out_ptr,
        blk_sz: triton.language.constexpr,
    ):
        pid = triton.language.program_id(0)
        blk = triton.language.arange(0, blk_sz)
        inp_blk = blk * vec_sz + pid
        msk = blk < bt_sz

        inp = triton.language.load(inp_ptr + inp_blk, msk, 0)

        if running_mean_ptr is not None and running_var_ptr is not None:
            mean = triton.language.load(running_mean_ptr + pid)
            var = triton.language.load(running_var_ptr + pid)
        else:
            mean = triton.language.sum(inp, 0) / bt_sz
            var = language.var(msk, inp, bt_sz, mean, corr=0)

        std = language.std(var, eps)
        out = (inp - mean) / std

        if wgt_ptr is not None:
            wgt = triton.language.load(wgt_ptr + pid)
            out = out * wgt

        if bis_ptr is not None:
            bis = triton.language.load(bis_ptr + pid)
            out = out + bis

        triton.language.store(out_ptr + inp_blk, out, msk)

    @staticmethod
    @triton.jit
    def backward(
        grad_out_ptr,
        inp_ptr,
        grad_inp_ptr,
        vec_sz,
        bt_sz,
        wgt_ptr,
        grad_wgt_ptr,
        grad_bis_ptr,
        eps,
        blk_sz: triton.language.constexpr,
    ):
        pid = triton.language.program_id(0)
        blk = triton.language.arange(0, blk_sz)
        inp_blk = blk * vec_sz + pid
        msk = blk < bt_sz

        grad_out = triton.language.load(grad_out_ptr + inp_blk, msk, 0)
        inp = triton.language.load(inp_ptr + inp_blk, msk, 0)
        wgt = triton.language.load(wgt_ptr + pid) if wgt_ptr is not None else 1

        mean = triton.language.sum(inp, 0) / bt_sz
        var = language.var(msk, inp, bt_sz, mean, corr=0)
        std = language.std(var, eps)

        xc = inp - mean
        xn = xc / std
        grad_xn = wgt * grad_out
        grad_std = -triton.language.sum((grad_xn * xc) / (std * std), axis=0)
        grad_var = 0.5 * grad_std / std
        grad_xc = grad_xn / std + (2.0 / bt_sz) * xc * grad_var
        grad_mu = triton.language.sum(triton.language.where(msk, grad_xc, 0.0))
        grad_inp = grad_xc - grad_mu / bt_sz

        triton.language.store(grad_inp_ptr + inp_blk, grad_inp, msk)

        if grad_wgt_ptr:
            grad_wgt = triton.language.sum(xn * grad_out, axis=0)
            triton.language.store(grad_wgt_ptr + pid, grad_wgt)

        if grad_bis_ptr:
            grad_bis = triton.language.sum(grad_out, axis=0)
            triton.language.store(grad_bis_ptr + pid, grad_bis)
