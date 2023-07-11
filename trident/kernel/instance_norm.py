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
        inp_ptr, num_ch, vec_sz, eps, out_ptr, vec_blk_sz: triton.language.constexpr, dtype: triton.language.constexpr
    ):
        pid = triton.language.program_id(0)
        bt = pid // num_ch
        ch = language.col(pid, num_ch)

        off = bt * (num_ch * vec_sz) + ch * vec_sz
        inp_ptr += off
        out_ptr += off

        mean = kernel.mean(inp_ptr, vec_sz, vec_blk_sz, dtype)
        var = kernel.variance(inp_ptr, vec_sz, mean, vec_blk_sz, dtype)
        std = language.std(var, eps)

        for off in range(0, vec_sz, vec_blk_sz):
            blk = triton.language.arange(0, vec_blk_sz) + off
            msk = blk < vec_sz

            inp = triton.language.load(inp_ptr + blk, msk, 0)
            out = language.norm(inp, mean, std)
            triton.language.store(out_ptr + blk, out, msk)
