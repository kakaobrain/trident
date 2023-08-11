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


class LeakyReLU:
    @staticmethod
    @triton.jit
    def forward(
        x_ptr,
        x_stride,
        y_ptr,
        y_stride,
        a,
        size_1,
        block_size: tl.constexpr,
    ):
        i = tl.program_id(0)
        j = tl.program_id(1)

        block = tl.arange(0, block_size) + j * block_size
        mask = block < size_1

        x_ptr += i * x_stride + block
        y_ptr += i * y_stride + block

        x = tl.load(x_ptr, mask, 0.0)
        y = language.leaky_relu(x, a)

        tl.store(y_ptr, y, mask)

    @staticmethod
    @triton.jit
    def backward(
        x_ptr,
        x_stride,
        d_ptr,
        d_stride,
        a,
        size,
        block_size: tl.constexpr,
    ):
        i = tl.program_id(0)
        j = tl.program_id(1)

        block = tl.arange(0, block_size) + j * block_size
        mask = block < size

        x_ptr += i * x_stride + block
        d_ptr += i * d_stride + block

        x = tl.load(x_ptr, mask, 0.0)
        d = tl.where(x > 0, 1, a)

        tl.store(d_ptr, d, mask)
