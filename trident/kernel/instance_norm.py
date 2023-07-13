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

from trident import kernel, language


class InstanceNorm:
    @staticmethod
    @triton.jit
    def forward(
        inp_ptr,
        num_ch,
        vec_sz,
        running_mean_ptr,
        running_var_ptr,
        wgt_ptr,
        bis_ptr,
        eps,
        out_ptr,
        blk_sz: triton.language.constexpr,
        dtype: triton.language.constexpr,
    ):
        pid = triton.language.program_id(0)
        bt = pid // num_ch
        ch = language.col(pid, num_ch)

        ptr_off = bt * num_ch * vec_sz + ch * vec_sz
        inp_ptr += ptr_off
        out_ptr += ptr_off

        if running_mean_ptr:
            mean = triton.language.load(running_mean_ptr + pid)
        else:
            mean = kernel.mean(inp_ptr, vec_sz, blk_sz, dtype)

        if running_var_ptr:
            var = triton.language.load(running_var_ptr + pid)
        else:
            var = kernel.variance(inp_ptr, vec_sz, mean, blk_sz, dtype)

        std = language.std(var, eps)

        for blk_off in range(0, vec_sz, blk_sz):
            blk, msk = language.make_block(vec_sz, blk_sz, blk_off)
            inp = triton.language.load(inp_ptr + blk, msk, 0)
            out = language.norm(inp, mean, std)

            if wgt_ptr:
                out *= triton.language.load(wgt_ptr + pid)

            if bis_ptr:
                out += triton.language.load(bis_ptr + pid)

            triton.language.store(out_ptr + blk, out, msk)

    @staticmethod
    @triton.jit
    def backward(
        grad_out_ptr,
        inp_ptr,
        grad_inp_ptr,
        num_ch,
        vec_sz,
        bt_sz,
        running_mean_ptr,
        running_var_ptr,
        wgt_ptr,
        grad_wgt_ptr,
        grad_bis_ptr,
        eps,
        blk_sz: triton.language.constexpr,
        dtype: triton.language.constexpr,
    ):
        pid = triton.language.program_id(0)
        bt = pid // num_ch
        ch = language.col(pid, num_ch)

        ptr_off = bt * num_ch * vec_sz + ch * vec_sz
        grad_out_ptr += ptr_off
        inp_ptr += ptr_off
        grad_inp_ptr += ptr_off

        if running_mean_ptr:
            mean = triton.language.load(running_mean_ptr + pid)
        else:
            mean = kernel.mean(inp_ptr, vec_sz, blk_sz, dtype)

        if running_var_ptr:
            var = triton.language.load(running_var_ptr + pid)
        else:
            var = kernel.variance(inp_ptr, vec_sz, mean, blk_sz, dtype)

        std = language.std(var, eps)

        for blk_off in range(0, vec_sz, blk_sz):
            blk, msk = language.make_block(vec_sz, blk_sz, blk_off)
            grad_out = triton.language.load(grad_out_ptr + blk, msk, 0)
            inp = triton.language.load(inp_ptr + blk, msk, 0)
            wgt = triton.language.load(wgt_ptr + pid) if wgt_ptr is not None else 1

            xc = inp - mean
            xn = xc / std
            grad_xn = wgt * grad_out
            grad_std = -triton.language.sum((grad_xn * xc) / (std * std), axis=0)
            grad_var = 0.5 * grad_std / std
            grad_xc = grad_xn / std + (2.0 / bt_sz) * xc * grad_var
            grad_mu = triton.language.sum(triton.language.where(msk, grad_xc, 0.0))
            grad_inp = grad_xc - grad_mu / bt_sz

            triton.language.store(grad_inp_ptr + blk, grad_inp, msk)

            if grad_wgt_ptr:
                grad_wgt = triton.language.sum(xn * grad_out, axis=0)
                triton.language.atomic_add(grad_wgt_ptr + pid, grad_wgt)

            if grad_bis_ptr:
                # 중간 텐서를 생성해서 보관해야함
                grad_bis = triton.language.sum(grad_out, axis=0)
                triton.language.atomic_add(grad_bis_ptr + pid, grad_bis)
