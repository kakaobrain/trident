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


class GELU:
    @staticmethod
    @triton.jit
    def forward(inp_ptr, inp_sz, out_ptr, blk_sz: triton.language.constexpr):
        pid = triton.language.program_id(0)
        blk = triton.language.arange(0, blk_sz) + pid * blk_sz
        msk = blk < inp_sz

        inp = triton.language.load(inp_ptr + blk, msk, 0)
        out = language.gelu(inp)

        triton.language.store(out_ptr + blk, out, msk)

    @staticmethod
    @triton.jit
    def backward(grad_out_ptr, inp_ptr, grad_inp_ptr, inp_sz, blk_sz: triton.language.constexpr):
        pid = triton.language.program_id(0)
        blk = triton.language.arange(0, blk_sz) + pid * blk_sz
        msk = blk < inp_sz

        inp = triton.language.load(inp_ptr + blk, msk, 0)
        a = 0.797884560802865
        b = language.tanh(a * (inp + 0.044715 * language.pow3(inp)))
        c = 1.0 + b
        d = inp * (1.0 - language.pow2(b)) * a * (1 + 0.134145 * language.pow2(inp))
        grad_out = triton.language.load(grad_out_ptr + blk, msk, 0)
        grad_inp = 0.5 * (c + d)

        triton.language.store(grad_inp_ptr + blk, grad_out * grad_inp, msk)
