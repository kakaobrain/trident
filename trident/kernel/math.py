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

from trident import language


@triton.jit
def mean_legacy(
    x_ptr,
    x_sz,
    blk_sz: tl.constexpr,
    dtype: tl.constexpr,
):
    res = tl.zeros([blk_sz], dtype)

    for off in range(0, x_sz, blk_sz):
        blk = tl.arange(0, blk_sz) + off
        msk = blk < x_sz

        num = tl.load(x_ptr + blk, msk, 0)
        res += num

    return tl.sum(res, 0) / x_sz


@triton.jit
def var_legacy(
    x_ptr,
    x_sz,
    mean,
    blk_sz: tl.constexpr,
    dtype: tl.constexpr,
):
    res = tl.zeros([blk_sz], tl.float32)

    for blk_off in range(0, x_sz, blk_sz):
        blk, msk = language.make_block(x_sz, blk_sz, blk_off)
        num = tl.load(x_ptr + blk, msk, 0).to(tl.float32)
        num = tl.where(msk, num - mean, 0)
        res += language.pow2(num)

    return (tl.sum(res, axis=0) / x_sz).to(dtype)
