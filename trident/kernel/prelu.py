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


class PReLU:
    @staticmethod
    @triton.jit
    def forward(x_ptr, x_stride, y_ptr, y_stride, w_ptr, size_1, block_size: triton.language.constexpr):
        i = triton.language.program_id(0)
        j = triton.language.program_id(1)

        block = triton.language.arange(0, block_size) + j * block_size
        mask = block < size_1

        x_ptr += i * x_stride + block
        y_ptr += i * y_stride + block
        w_ptr += triton.language.arange(0, block_size)

        x = triton.language.load(x_ptr, mask, 0.0)
        w = triton.language.load(w_ptr, mask, 0.0)
        y = language.leaky_relu(x, w)

        triton.language.store(y_ptr, y, mask)

    @staticmethod
    @triton.jit
    def backward(x_ptr, x_stride, dx_ptr, dx_stride, w_ptr, dw_ptr, size, block_size: triton.language.constexpr):
        i = triton.language.program_id(0)
        j = triton.language.program_id(1)

        block = triton.language.arange(0, block_size) + j * block_size
        mask = block < size

        x_ptr += i * x_stride + block
        dx_ptr += i * dx_stride + block

        w_ptr += triton.language.arange(0, block_size)
        dw_ptr += i * x_stride + block

        x = triton.language.load(x_ptr, mask, 0.0)
        w = triton.language.load(w_ptr, mask, 0.0)

        dx = triton.language.where(x > 0, 1, w)
        dw = triton.language.where(x > 0, 0, x)

        triton.language.store(dx_ptr, dx, mask)
        triton.language.store(dw_ptr, dw, mask)
