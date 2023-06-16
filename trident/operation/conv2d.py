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

import torch
import triton

from trident import kernel


class Conv2d(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        input, weight, bias = args

        assert input.is_cuda & input.is_contiguous()
        assert weight.is_cuda & weight.is_contiguous()

        if bias is not None:
            assert bias.is_cuda & bias.is_contiguous()

        num_batches, in_channels, in_height, in_width = input.shape
        out_channels, _, wt_height, wt_width = weight.shape
        out_height = Conv2d.__get_out_size(in_height, wt_height, 1)
        out_width = Conv2d.__get_out_size(in_width, wt_width, 1)

        output = torch.zeros(num_batches, out_channels, out_height, out_width, dtype=torch.float, device='cuda')

        assert output.is_cuda & output.is_contiguous()

        def grid(meta):
            return (num_batches * out_channels * out_height * out_width,)

        kernel.Conv2d.forward[grid](input, in_channels, in_height, in_width, input.stride(0), input.stride(1),
                                    weight, wt_width, wt_height, weight.stride(0), weight.stride(1), bias,
                                    output, out_channels, out_height, out_width,
                                    output.stride(0), output.stride(1), output.stride(2),
                                    channel_block_size=triton.next_power_of_2(in_channels),
                                    weight_block_size=triton.next_power_of_2(wt_width))

        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def __get_out_size(in_size, wt_size, stride):
        return ((in_size - wt_size) // stride) + 1
