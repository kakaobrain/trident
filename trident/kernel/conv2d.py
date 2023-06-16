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


class Conv2d:
    @staticmethod
    @triton.jit
    def forward(input_ptr, in_channels, in_height, in_width, in_batch_stride, in_channel_stride,
                weight_ptr, wt_height, wt_width, wt_batch_stride, wt_channel_stride, bias_ptr,
                output_ptr, out_channels, out_height, out_width, out_batch_stride, out_channel_stride, out_width_stride,
                channel_block_size: triton.language.constexpr, weight_block_size: triton.language.constexpr):
        program_id = triton.language.program_id(0)

        in_batch = language.batch(program_id, out_channels, out_height, out_width)
        out_channel = language.channel(program_id, out_channels, out_height, out_width)
        out_height = language.row(program_id, out_height, out_width)
        out_width = language.col(program_id, out_width)

        channel_block = triton.language.arange(0, channel_block_size)
        channel_mask = channel_block < in_channels

        input_block = triton.language.arange(0, weight_block_size)
        input_block = input_block[:, None] * in_width + input_block[None, :]
        input_block = channel_block[:, None, None] * in_channel_stride + input_block[None, :, :]

        weight_block = triton.language.arange(0, weight_block_size)
        weight_block = weight_block[:, None] * wt_width + weight_block[None, :]
        weight_block = channel_block[:, None, None] * wt_channel_stride + weight_block[None, :, :]

        mask = triton.language.arange(0, weight_block_size) < wt_height
        mask = mask[:, None] & mask[None, :]
        mask = channel_mask[:, None, None] & mask[None, :, :]

        input_ptr += in_batch * in_batch_stride + out_height * in_height + out_width
        input = triton.language.load(input_ptr + input_block, mask, 0.0)

        weight_ptr += out_channel * wt_batch_stride
        weight = triton.language.load(weight_ptr + weight_block, mask, 0.0)

        output = language.sum(input * weight)

        if bias_ptr:
            output += triton.language.load(bias_ptr + out_channel)

        output_ptr += in_batch * out_batch_stride + out_channel * out_channel_stride + out_height * out_width_stride + out_width
        triton.language.store(output_ptr, output)
