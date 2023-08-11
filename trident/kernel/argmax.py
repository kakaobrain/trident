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


class Argmax:
    @staticmethod
    @triton.jit
    def forward(
        output_ptr,
        input_ptr,
        y_size,
        x_size,
        dim: tl.constexpr,
        block_size: tl.constexpr,
        dtype: tl.constexpr,
    ):
        offset = tl.program_id(0)
        maximum, output = language.max(input_ptr, y_size, x_size, offset, dim, block_size, dtype)
        output_block_ptr = tl.make_block_ptr(
            output_ptr,
            shape=(y_size if dim == 0 else x_size,),
            strides=(1,),
            offsets=(offset,),
            block_shape=(1,),
            order=(0,),
        )
        tl.store(output_block_ptr, output)
