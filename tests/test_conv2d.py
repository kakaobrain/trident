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

import pytest
import torch

import trident
from tests import util


@pytest.mark.parametrize('num_batches, in_channels, out_channels', [(1, 3, 1), (5, 4, 8), (6, 3, 9), (10, 16, 36)])
def test_function(num_batches, in_channels, out_channels):
    input = torch.randn(num_batches, in_channels, 5, 5, dtype=torch.float, device='cuda')
    weight = torch.randn(out_channels, in_channels, 3, 3, dtype=torch.float, device='cuda')
    bias = torch.randn(out_channels, dtype=torch.float, device='cuda')

    assert util.equal(torch.nn.functional.conv2d(input, weight, bias), trident.function.conv2d(input, weight, bias))
    assert util.equal(torch.nn.functional.conv2d(input, weight), trident.function.conv2d(input, weight))


@pytest.mark.parametrize('in_channels, out_channels, kernel_size', [(1, 1, 3), (3, 16, 4), (4, 32, 5), (4, 4, 32)])
def test_forward(in_channels, out_channels, kernel_size):
    conv2d_tc = torch.nn.Conv2d(in_channels, out_channels, kernel_size, dtype=torch.float, device='cuda')
    conv2d_td = trident.Conv2d(in_channels, out_channels, kernel_size)
    conv2d_td.weight, conv2d_td.bias = conv2d_tc.weight, conv2d_tc.bias

    input = torch.randn(4, in_channels, 64, 64, dtype=torch.float, device='cuda')

    assert util.equal(conv2d_tc.forward(input), conv2d_td.forward(input))
