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

from trident import kernel, language


class InstanceNorm:
    @staticmethod
    @triton.jit
    def forward(
        p_inp,
        num_ch,
        vec_sz,
        p_run_mean,
        p_run_var,
        p_wgt,
        p_bis,
        eps,
        p_out,
        blk_sz: tl.constexpr,
        dtype: tl.constexpr,
    ):
        pid = tl.program_id(0)
        bt = pid // num_ch
        ch = language.col(pid, num_ch)

        ptr_off = bt * num_ch * vec_sz + ch * vec_sz
        p_inp += ptr_off
        p_out += ptr_off

        if p_run_mean is None:
            mean = kernel.mean_legacy(p_inp, vec_sz, blk_sz, dtype)
        else:
            mean = tl.load(p_run_mean + ch)

        if p_run_var is None:
            var = kernel.var_legacy(p_inp, vec_sz, mean, blk_sz, dtype)
        else:
            var = tl.load(p_run_var + ch)

        std = language.std(var, eps)

        for blk_off in range(0, vec_sz, blk_sz):
            blk, msk = language.make_block(vec_sz, blk_sz, blk_off)
            inp = tl.load(p_inp + blk, msk, 0)
            out = language.norm(inp, mean, std)

            if p_wgt is not None:
                out *= tl.load(p_wgt + ch)

            if p_bis is not None:
                out += tl.load(p_bis + ch)

            tl.store(p_out + blk, out, msk)

    @staticmethod
    @triton.jit
    def backward(
        p_grad_out,
        p_inp,
        p_grad_inp,
        num_ch,
        vec_sz,
        wgt_ptr,
        p_stg_grad_wgt,
        p_stg_grad_bis,
        eps,
        blk_sz: tl.constexpr,
    ):
        pid = tl.program_id(0)
        bt = pid // num_ch
        ch = language.col(pid, num_ch)

        ptr_off = bt * num_ch * vec_sz + ch * vec_sz
        p_grad_out += ptr_off
        p_inp += ptr_off
        p_grad_inp += ptr_off

        blk, msk = language.make_block(vec_sz, blk_sz)
        inp = tl.load(p_inp + blk, msk, 0)
        wgt = tl.load(wgt_ptr + ch) if wgt_ptr is not None else 1
        mean = tl.sum(inp, 0) / vec_sz
        var = tl.sum(language.pow2(tl.where(msk, inp - mean, 0.0)), 0) / vec_sz
        std = language.std(var, eps)
        mean_ctr = tl.where(msk, inp - mean, 0)
        norm = mean_ctr / std

        grad_out = tl.load(p_grad_out + blk, msk, 0)
        grad_norm = wgt * grad_out
        grad_std = ((grad_norm * mean_ctr) / -language.pow2(std)) / (2 * std)
        grad_var = tl.sum(grad_std, 0) / vec_sz
        grad_dist = 2 * mean_ctr * grad_var
        grad_mean_ctr = tl.where(msk, (grad_norm / std) + grad_dist, 0)
        grad_mean = -tl.sum(grad_mean_ctr, 0) / vec_sz
        grad_inp = grad_mean_ctr + grad_mean
        tl.store(p_grad_inp + blk, grad_inp, msk)

        if p_stg_grad_wgt is not None:
            grad_wgt = tl.sum(norm * grad_out, 0)
            ptr_off = bt * num_ch + ch
            tl.store(p_stg_grad_wgt + ptr_off, grad_wgt)

        if p_stg_grad_bis is not None:
            grad_bis = tl.sum(grad_out, 0)
            ptr_off = bt * num_ch + ch
            tl.store(p_stg_grad_bis + ptr_off, grad_bis)

    @staticmethod
    @triton.jit
    def mean_var(
        p_inp,
        num_bt,
        num_ch,
        vec_sz,
        p_mean,
        p_var,
        blk_sz: tl.constexpr,
        dtype: tl.constexpr,
    ):
        pid = tl.program_id(0)
        bt = pid // num_ch
        ch = language.col(pid, num_ch)

        ptr_off = bt * num_ch * vec_sz + ch * vec_sz
        p_inp += ptr_off

        mean = kernel.mean_legacy(p_inp, vec_sz, blk_sz, dtype)
        var = kernel.var_legacy(p_inp, vec_sz, mean, blk_sz, dtype)

        tl.atomic_add(p_mean + ch, mean / num_bt)
        tl.atomic_add(p_var + ch, var / num_bt)

    @staticmethod
    @triton.jit
    def optimize(
        p_mean,
        p_run_mean,
        p_var,
        p_run_var,
        num_ch,
        momentum,
        blk_size: tl.constexpr,
    ):
        blk, msk = language.make_block(num_ch, blk_size)

        mean = tl.load(p_mean + blk, msk, 0)
        var = tl.load(p_var + blk, msk, 0)
        run_mean = tl.load(p_run_mean + blk, msk, 0)
        run_var = tl.load(p_run_var + blk, msk, 0)
        run_mean = mean * momentum + run_mean * (1 - momentum)
        run_var = var * momentum + run_var * (1 - momentum)

        tl.store(p_run_mean + blk, run_mean, msk)
        tl.store(p_run_var + blk, run_var, msk)
