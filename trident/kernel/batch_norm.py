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
import triton.language as tl

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
        blk_sz: tl.constexpr,
    ):
        pid = tl.program_id(0)
        blk = tl.arange(0, blk_sz)
        inp_blk = blk * vec_sz + pid
        msk = blk < bt_sz

        inp = tl.load(inp_ptr + inp_blk, msk, 0)

        if running_mean_ptr is not None and running_var_ptr is not None:
            mean = tl.load(running_mean_ptr + pid)
            var = tl.load(running_var_ptr + pid)
        else:
            mean = tl.sum(inp, 0) / bt_sz
            centered_mean = tl.where(msk, inp - mean, 0.0)
            var = tl.sum(centered_mean * centered_mean, 0) / bt_sz

        std = tl.sqrt(var + eps)
        out = (inp - mean) / std

        if wgt_ptr is not None:
            wgt = tl.load(wgt_ptr + pid)
            out = out * wgt

        if bis_ptr is not None:
            bis = tl.load(bis_ptr + pid)
            out = out + bis

        tl.store(out_ptr + inp_blk, out, msk)

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
        blk_sz: tl.constexpr,
    ):
        pid = tl.program_id(0)
        blk = tl.arange(0, blk_sz)
        inp_blk = blk * vec_sz + pid
        msk = blk < bt_sz

        grad_out = tl.load(grad_out_ptr + inp_blk, msk, 0)
        inp = tl.load(inp_ptr + inp_blk, msk, 0)
        wgt = tl.load(wgt_ptr + pid) if wgt_ptr is not None else 1

        mean = tl.sum(inp, 0) / bt_sz
        centered_mean = tl.where(msk, inp - mean, 0.0)
        var = tl.sum(centered_mean * centered_mean, 0) / bt_sz
        std = tl.sqrt(var + eps)

        xc = inp - mean
        xn = xc / std
        grad_xn = wgt * grad_out
        grad_std = -tl.sum((grad_xn * xc) / (std * std), axis=0)
        grad_var = 0.5 * grad_std / std
        grad_xc = grad_xn / std + (2.0 / bt_sz) * xc * grad_var
        grad_mu = tl.sum(tl.where(msk, grad_xc, 0.0))
        grad_inp = grad_xc - grad_mu / bt_sz

        tl.store(grad_inp_ptr + inp_blk, grad_inp, msk)

        if grad_wgt_ptr:
            grad_wgt = tl.sum(xn * grad_out, axis=0)
            tl.store(grad_wgt_ptr + pid, grad_wgt)

        if grad_bis_ptr:
            grad_bis = tl.sum(grad_out, axis=0)
            tl.store(grad_bis_ptr + pid, grad_bis)
