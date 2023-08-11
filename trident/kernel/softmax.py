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


class Softmax:
    @staticmethod
    @triton.jit
    def forward(
        inp_ptr,
        y_size,
        vec_sz,
        out_ptr,
        blk_sz: tl.constexpr,
        dtype: tl.constexpr,
    ):
        pid = tl.program_id(0)
        max = language.max(
            inp_ptr,
            y_size,
            vec_sz,
            pid,
            language.dim[1],
            blk_sz,
            dtype,
        )
        acc = 0.0
        off = pid * vec_sz
        inp_ptr += off
        out_ptr += off

        for blk_off in range(0, vec_sz, blk_sz):
            blk, msk = language.make_block(vec_sz, blk_sz, blk_off)
            inp = tl.load(inp_ptr + blk, msk, -float("inf"))
            num = language.exp(inp - max)
            acc += tl.sum(num, 0)

        for blk_off in range(0, vec_sz, blk_sz):
            blk, msk = language.make_block(vec_sz, blk_sz, blk_off)
            inp = tl.load(inp_ptr + blk, msk, -float("inf"))
            out = language.exp(inp - max) / acc
            tl.store(out_ptr + blk, out, msk)

    @staticmethod
    @triton.jit
    def backward(
        grad_out_ptr,
        out_ptr,
        vec_sz,
        grad_inp_ptr,
        blk_sz: tl.constexpr,
    ):
        pid = tl.program_id(0)
        blk, msk = language.make_block(vec_sz, blk_sz, 0)
        blk = blk + pid * vec_sz

        grad_out = tl.load(grad_out_ptr + blk, msk, 0)
        out = tl.load(out_ptr + blk, msk, 0)
        grad_inp = language.diagflat(out, blk_sz) - (out[:, None] * out[None, :])
        grad_inp = language.gemv(grad_inp, grad_out)

        tl.store(grad_inp_ptr + blk, grad_inp, msk)
