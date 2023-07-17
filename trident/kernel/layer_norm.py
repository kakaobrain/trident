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


class LayerNorm:
    @staticmethod
    @triton.jit
    def forward(
        inp_ptr,
        vec_sz,
        wgt_ptr,
        bis_ptr,
        eps,
        out_ptr,
        blk_sz: triton.language.constexpr,
        dtype: triton.language.constexpr,
    ):
        pid = triton.language.program_id(0)
        off = pid * vec_sz

        inp_ptr += off
        out_ptr += off

        mean = kernel.mean(inp_ptr, vec_sz, blk_sz, dtype)
        var = kernel.var(inp_ptr, vec_sz, mean, blk_sz, dtype)
        std = language.std(var, eps)

        for blk_off in range(0, vec_sz, blk_sz):
            blk, msk = language.make_block(vec_sz, blk_sz, blk_off)
            inp = triton.language.load(inp_ptr + blk, msk, 0)
            out = language.norm(inp, mean, std)

            if wgt_ptr:
                wgt = triton.language.load(wgt_ptr + blk, msk, 0)
                out *= wgt

            if bis_ptr:
                bis = triton.language.load(bis_ptr + blk, msk, 0)
                out += bis

            triton.language.store(out_ptr + blk, out, msk)

    @staticmethod
    @triton.jit
    def backward(
        grad_out_ptr,
        inp_ptr,
        grad_inp_ptr,
        vec_sz,
        wgt_ptr,
        grad_wgt_ptr,
        grad_bis_ptr,
        eps,
        blk_sz: triton.language.constexpr,
        dtype: triton.language.constexpr,
    ):
        pid = triton.language.program_id(0)
        blk_off = pid * vec_sz

        grad_out_ptr += blk_off
        inp_ptr += blk_off
        grad_inp_ptr += blk_off

        mean = kernel.mean(inp_ptr, vec_sz, blk_sz, dtype)
        var = kernel.var(inp_ptr, vec_sz, mean, blk_sz, dtype)
        std = language.std(var, eps)

        for blk_off in range(0, vec_sz, blk_sz):
            blk, msk = language.make_block(vec_sz, blk_sz, blk_off)
            grad_out = triton.language.load(grad_out_ptr + blk, msk, 0)
            inp = triton.language.load(inp_ptr + blk, msk, 0)
            wgt = triton.language.load(wgt_ptr + blk, msk, 0)

            a = grad_out * wgt
            b = (inp - mean) / std
            b = triton.language.where(msk, b, 0)
            c = language.gemv(b, a) / vec_sz
            d = language.gemv(grad_out, wgt) / vec_sz
            grad_inp = (a - c * b - d) / std

            triton.language.store(grad_inp_ptr + blk, grad_inp, msk)

            if grad_wgt_ptr:
                triton.language.atomic_add(grad_wgt_ptr + blk, grad_out * b, msk)

            if grad_bis_ptr:
                triton.language.atomic_add(grad_bis_ptr + blk, grad_out, msk)
