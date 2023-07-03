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


def get_configs_linear_io_bound():
    configs = []
    for block_size_n in [64, 128]:
        for num_stages in [2, 3]:
            for num_warps in [2, 4]:
                configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': block_size_n},
                                             num_stages=num_stages, num_warps=num_warps))
    return configs


@triton.autotune(
    configs=get_configs_linear_io_bound(),
    key=['size_m', 'size_n']
)
@triton.jit
def linear(x_ptr, stride_x_m, stride_x_k,
           y_ptr, stride_y_m, stride_y_n,
           w_ptr, stride_w_n, stride_w_k,
           b_ptr, stride_b_n,
           size_m, size_k, size_n,
           ACTIVATION: tl.constexpr,
           BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    i = tl.program_id(0)
    j = tl.program_id(1)

    range_m = tl.arange(0, BLOCK_SIZE_M) + i * BLOCK_SIZE_M
    range_k = tl.arange(0, BLOCK_SIZE_K)
    range_n = tl.arange(0, BLOCK_SIZE_N) + j * BLOCK_SIZE_N

    total = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    x_ptr += range_m[:, None] * stride_x_m + range_k[None, :] * stride_x_k
    w_ptr += range_n[None, :] * stride_w_n + range_k[:, None] * stride_w_k

    for k in range(0, size_k, BLOCK_SIZE_K):
        mask_m = range_m[:, None] < size_m
        mask_n = range_n[None, :] < size_n

        x = tl.load(x_ptr, mask_m & (range_k[None, :] + k < size_k), 0.0)
        w = tl.load(w_ptr, mask_n & (range_k[:, None] + k < size_k), 0.0)
        total += tl.dot(x, w, False)

        x_ptr += BLOCK_SIZE_K * stride_x_k
        w_ptr += BLOCK_SIZE_K * stride_w_k

    if b_ptr is not None:
        b_ptr += range_n * stride_b_n
        mask_n = range_n < size_n
        b = tl.load(b_ptr, mask_n, 0.0)
        total += b[None, :]

    if ACTIVATION == 'relu':
        total = language.relu(total)
    elif ACTIVATION == 'leaky_relu':
        total = language.leaky_relu(total, 1e-2)

    y_ptr += range_m[:, None] * stride_y_m + range_n[None, :] * stride_y_n
    mask_m = range_m[:, None] < size_m
    mask_n = range_n[None, :] < size_n
    tl.store(y_ptr, total, mask_m & mask_n)
