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


class MaxPool2D:
    @staticmethod
    @triton.jit
    def forward(x_ptr, x_batch_stride, x_channel_stride, x_row_stride,
                y_ptr, y_batch_stride, y_channel_stride, y_row_stride,
                num_channels, num_rows, num_cols,
                kernel_size: triton.language.constexpr):
        program_id = triton.language.program_id(0)

        num_blocks_per_row = num_rows // kernel_size
        num_blocks_per_channel = num_blocks_per_row
        num_blocks_per_batch = num_channels * num_blocks_per_channel
        num_kernels_per_col = num_cols // kernel_size

        batch = program_id // num_blocks_per_batch
        channel = (program_id % num_blocks_per_batch) // num_blocks_per_channel
        row = program_id % num_blocks_per_row

        x_ptr += batch * x_batch_stride + channel * x_channel_stride
        x_ptr += row * kernel_size * x_row_stride
        y_ptr += batch * y_batch_stride + channel * y_channel_stride
        y_ptr += row * y_row_stride

        block = triton.language.arange(0, kernel_size)
        block = block[:, None] * x_row_stride + block[None, :]
        block = triton.language.ravel(block)

        for i in range(0, num_kernels_per_col):
            x = triton.language.load(x_ptr + block)
            y = language.max1d(x)
            triton.language.store(y_ptr + i, y)
            block += kernel_size
