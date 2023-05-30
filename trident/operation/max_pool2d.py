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

import torch

import trident.kernel


class MaxPool2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        x, kernel_size = args

        assert x.is_cuda and x.is_contiguous()

        num_batches, num_channels, num_rows, num_cols = x.shape
        num_row_blocks = MaxPool2d.__get_out_size(num_rows, kernel_size)
        num_col_blocks = MaxPool2d.__get_out_size(num_cols, kernel_size)

        y = torch.empty(num_batches, num_channels, num_row_blocks, num_col_blocks, device='cuda')

        assert y.is_contiguous()

        block_size = max(num_cols // kernel_size // 4, 1)
        grid = lambda meta: (num_batches * num_channels * num_row_blocks * num_col_blocks // block_size,)
        trident.kernel.MaxPool2D.forward[grid](x, x.stride(0), x.stride(1), x.stride(2),
                                               y, y.stride(0), y.stride(1), y.stride(2),
                                               num_channels, num_rows, num_cols,
                                               kernel_size=kernel_size, block_size=block_size)

        return y

    @staticmethod
    def __get_out_size(num_elements, kernel_size):
        return ((num_elements - (kernel_size - 1) - 1) // kernel_size) + 1
