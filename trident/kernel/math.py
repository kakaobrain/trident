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


@triton.jit
def mean(x_ptr, x_sz, blk_sz: triton.language.constexpr, dtype: triton.language.constexpr):
    mem = triton.language.zeros([blk_sz], dtype)

    for off in range(0, x_sz, blk_sz):
        blk = triton.language.arange(0, blk_sz) + off
        msk = blk < x_sz

        num = triton.language.load(x_ptr + blk, msk, 0)
        mem += num

    return triton.language.sum(mem, 0) / x_sz


@triton.jit
def var(x_ptr, x_sz, mean, blk_sz: triton.language.constexpr, dtype: triton.language.constexpr):
    mem = triton.language.zeros([blk_sz], triton.language.float32)

    for off in range(0, x_sz, blk_sz):
        blk = triton.language.arange(0, blk_sz) + off
        msk = blk < x_sz

        num = triton.language.load(x_ptr + blk, msk, 0).to(triton.language.float32)
        num = triton.language.where(msk, num - mean, 0)
        mem += language.pow2(num)

    return (triton.language.sum(mem, axis=0) / x_sz).to(dtype)
