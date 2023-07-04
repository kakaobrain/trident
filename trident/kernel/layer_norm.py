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


class LayerNorm:
    @staticmethod
    @triton.jit
    def forward(inp_ptr, vec_sz, wgt_ptr, bis_ptr, eps, mean_ptr, std_ptr, out_ptr,
                vec_bs: triton.language.constexpr):
        pid = triton.language.program_id(0)
        wgt_blk = triton.language.arange(0, vec_bs)
        inp_blk = wgt_blk + pid * vec_sz
        msk = wgt_blk < vec_sz

        inp = triton.language.load(inp_ptr + inp_blk, msk, 0)
        mean = language.sum(inp) / vec_sz
        var = language.pow2(triton.language.where(msk, inp - mean, 0))
        var = language.sum(var) / vec_sz
        std = triton.language.sqrt(var + eps)
        out = (inp - mean) / std

        if wgt_ptr:
            wgt = triton.language.load(wgt_ptr + wgt_blk, msk, 0)
            out = out * wgt

        if bis_ptr:
            bis = triton.language.load(bis_ptr + wgt_blk, msk, 0)
            out = out + bis

        if mean_ptr:
            triton.language.store(mean_ptr + pid, mean)

        if std_ptr:
            triton.language.store(std_ptr + pid, std)

        triton.language.store(out_ptr + inp_blk, out, msk)

    @staticmethod
    @triton.jit
    def backward(grad_out_ptr, inp_ptr, grad_inp_ptr, vec_sz, wgt_ptr, grad_wgt_ptr, grad_bis_ptr, mean_ptr,
                 std_ptr, vec_bs: triton.language.constexpr):
        pid = triton.language.program_id(0)
        wgt_blk = triton.language.arange(0, vec_bs)
        inp_blk = wgt_blk + pid * vec_sz
        msk = wgt_blk < vec_sz

        grad_out = triton.language.load(grad_out_ptr + inp_blk, msk, 0)
        inp = triton.language.load(inp_ptr + inp_blk, msk, 0)
        mean = triton.language.load(mean_ptr + pid)
        std = triton.language.load(std_ptr + pid)
        wgt = triton.language.load(wgt_ptr + wgt_blk, msk, 0)

        a = grad_out * wgt
        b = (inp - mean) / std
        b = triton.language.where(msk, b, 0)
        c = language.gemv(b, a) / vec_sz
        d = language.gemv(grad_out, wgt) / vec_sz
        grad_inp = (a - c * b - d) / std

        triton.language.store(grad_inp_ptr + inp_blk, grad_inp, msk)

        if grad_wgt_ptr:
            grad_wgt = grad_out * b
            triton.language.atomic_add(grad_wgt_ptr + wgt_blk, grad_wgt, msk)

        if grad_bis_ptr:
            grad_bis = grad_out
            triton.language.atomic_add(grad_bis_ptr + wgt_blk, grad_bis, msk)
