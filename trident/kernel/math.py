"""
Copyright 2023 ⓒ Kakao Brain Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import triton
import triton.language as tl


@triton.jit
def softmax_forward(x_ptr, stride_xm, stride_xn, y_ptr, stride_ym, stride_yn, size_m, size_n,
                    BLOCK_SIZE_N: tl.constexpr):
    i = tl.program_id(0)

    block_n = tl.arange(0, BLOCK_SIZE_N)
    mask_x = block_n < size_n

    x_ptr += i * stride_xm + block_n * stride_xn
    x = tl.load(x_ptr, mask_x, -float('inf'))

    numerator = tl.exp(x - tl.max(x, 0))
    y = numerator / tl.sum(numerator, 0)

    y_ptr += i * stride_ym + block_n * stride_yn
    tl.store(y_ptr, y, mask_x)


@triton.jit
def softmax_backward(y_ptr, stride_ym, stride_yn, t_ptr, stride_tm, stride_tn, d_ptr, stride_dm, stride_dn, size_n,
                     BLOCK_SIZE_N: tl.constexpr):
    i = tl.program_id(0)

    block_n = tl.arange(0, BLOCK_SIZE_N) + i * BLOCK_SIZE_N
    mask = block_n < size_n

    y_ptr += i * stride_ym + block_n * stride_yn
    y = tl.load(y_ptr, mask, 0.0)

    t_ptr += i * stride_tm + block_n * stride_tn
    t = tl.load(t_ptr, mask, 0.0)

    d = y - t

    d_ptr += i * stride_dm + block_n * stride_dn
    tl.store(d_ptr, d, mask)
