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


class Softmax:
    @staticmethod
    @triton.jit
    def forward(inp_ptr, vec_sz, vec_st, out_ptr,
                vec_bs: triton.language.constexpr, dtype: triton.language.constexpr):
        pid = triton.language.program_id(0)
        blk = triton.language.arange(0, vec_bs)
        msk = blk < vec_sz
        blk = blk + pid * vec_st

        inp = triton.language.load(inp_ptr + blk, msk, -float('inf'))
        numer = language.exp(inp - triton.language.max(inp, 0), dtype)
        denom = triton.language.sum(numer, 0)
        out = numer / denom

        triton.language.store(out_ptr + blk, out, msk)
