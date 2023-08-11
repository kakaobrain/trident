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


class ReLU:
    @staticmethod
    @triton.jit
    def forward(x_ptr, y_ptr, stride, size, block_size: tl.constexpr):
        i = tl.program_id(0)
        j = tl.program_id(1)

        block = tl.arange(0, block_size) + j * block_size
        mask = block < size
        span = i * stride + block

        x = tl.load(x_ptr + span, mask, 0.0)
        y = language.relu(x)

        tl.store(y_ptr + span, y, mask)

    @staticmethod
    @triton.jit
    def backward(dx_ptr, x_ptr, stride, size, block_size: tl.constexpr):
        i = tl.program_id(0)
        j = tl.program_id(1)

        block = tl.arange(0, block_size) + j * block_size
        mask = block < size
        span = i * stride + block

        x = tl.load(x_ptr + span, mask, 0.0)
        grad_x = tl.where(x > 0, 1, 0)

        tl.store(dx_ptr + span, grad_x, mask)
