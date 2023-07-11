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


class Dropout:
    @staticmethod
    @triton.jit
    def forward(inp_ptr, inp_sz, p, seed, out_ptr, inp_bs: triton.language.constexpr):
        pid = triton.language.program_id(0)
        blk = triton.language.arange(0, inp_bs) + pid * inp_bs
        msk = blk < inp_sz

        inp = triton.language.load(inp_ptr + blk, msk)
        rnd = triton.language.rand(seed, blk)
        out = triton.language.where(rnd > p, inp / (1.0 - p), 0.0)

        triton.language.store(out_ptr + blk, out, msk)

    @staticmethod
    @triton.jit
    def backward(
        grad_out_ptr,
        out_ptr,
        out_sz,
        grad_inp_ptr,
        out_bs: triton.language.constexpr,
    ):
        pid = triton.language.program_id(0)
        blk = triton.language.arange(0, out_bs) + pid * out_bs
        msk = blk < out_sz

        grad_out = triton.language.load(grad_out_ptr + blk, msk)
        out = triton.language.load(out_ptr + blk, msk)
        out = triton.language.abs(out)
        grad_inp = triton.language.where(out > 0.0, 1.0, 0.0)

        triton.language.store(grad_inp_ptr + blk, grad_out * grad_inp, msk)
