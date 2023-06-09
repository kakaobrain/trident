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

from trident import language


class MaxPool2d:
    @staticmethod
    @triton.jit
    def forward(x_ptr, x_batch_stride, x_channel_stride, x_row_stride,
                y_ptr, y_batch_stride, y_channel_stride, y_row_stride,
                num_channels, num_rows, num_cols,
                kernel_size: triton.language.constexpr, block_size: triton.language.constexpr):
        program_id = triton.language.program_id(0)

        num_row_kernels = num_rows // kernel_size
        num_col_kernels = num_cols // kernel_size
        num_col_blocks = num_col_kernels // block_size

        batch = language.batch(program_id, num_channels, num_row_kernels, num_col_blocks)
        channel = language.channel(program_id, num_channels, num_row_kernels, num_col_blocks)
        row_block = language.row(program_id, num_row_kernels, num_col_blocks)
        col_block = language.col(program_id, num_col_blocks)

        x_row_block_stride = kernel_size * x_row_stride
        x_col_block_stride = block_size * kernel_size
        y_row_block_stride = y_row_stride
        y_col_block_stride = block_size * col_block

        x_ptr += batch * x_batch_stride + channel * x_channel_stride
        x_ptr += row_block * x_row_block_stride + col_block * x_col_block_stride
        y_ptr += batch * y_batch_stride + channel * y_channel_stride
        y_ptr += row_block * y_row_block_stride + y_col_block_stride

        x_row_block = triton.language.arange(0, kernel_size)
        x_row_block = triton.language.ravel(x_row_block[:, None] * x_row_stride + x_row_block[None, :])
        x_col_block = triton.language.arange(0, block_size) * kernel_size
        x_block = x_row_block[:, None] + x_col_block[None, :]

        x = triton.language.load(x_ptr + x_block)
        y = triton.language.max(x, axis=0)
        y_block = triton.language.arange(0, block_size)

        triton.language.store(y_ptr + y_block, y)
