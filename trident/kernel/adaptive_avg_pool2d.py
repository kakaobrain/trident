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


class AdaptiveAvgPool2d:
    @staticmethod
    @triton.jit
    def forward(
        x_ptr,
        x_batch_stride,
        x_channel_stride,
        x_row_stride,
        y_ptr,
        y_batch_stride,
        y_channel_stride,
        y_row_stride,
        num_channels,
        num_rows,
        num_cols,
        output_size,
        kernel_size: triton.language.constexpr,
        block_size: triton.language.constexpr,
    ):
        program_id = triton.language.program_id(0)

        num_blocks = output_size // block_size

        batch = language.batch(program_id, num_channels, output_size, num_blocks)
        channel = language.channel(program_id, num_channels, output_size, num_blocks)
        row = language.row(program_id, output_size, num_blocks)
        block = language.col(program_id, num_blocks)
        col = block * block_size

        row_offset = (row * (num_rows / output_size)).to(triton.language.int32)
        col_offsets = (triton.language.arange(0, block_size) + col) * (
            num_cols / output_size
        )
        col_offsets = col_offsets.to(triton.language.int32)

        x_ptr += batch * x_batch_stride + channel * x_channel_stride
        x_ptr += row_offset * x_row_stride
        y_ptr += batch * y_batch_stride + channel * y_channel_stride
        y_ptr += row * y_row_stride + col

        kernel_block = triton.language.arange(0, kernel_size)
        kernel_block = triton.language.ravel(
            kernel_block[:, None] * x_row_stride + kernel_block[None, :]
        )
        x_block = kernel_block[:, None] + col_offsets[None, :]

        x = triton.language.load(x_ptr + x_block)
        y = triton.language.sum(x, axis=0) / (kernel_size * kernel_size)
        y_block = triton.language.arange(0, block_size)

        triton.language.store(y_ptr + y_block, y)
