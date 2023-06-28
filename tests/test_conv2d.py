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


@pytest.mark.parametrize('num_bt, inp_ch, out_ch', [(1, 3, 1), (5, 4, 8), (6, 3, 9), (10, 16, 36)])
def test_function(num_bt, inp_ch, out_ch, device):
    inp = torch.randn(num_bt, inp_ch, 5, 5, device=device)
    wgt = torch.randn(out_ch, inp_ch, 2, 2, device=device)
    bis = torch.randn(out_ch, device=device)

    assert util.equal(torch.nn.functional.conv2d(inp, wgt, bis), trident.function.conv2d(inp, wgt, bis))
    assert util.equal(torch.nn.functional.conv2d(inp, wgt), trident.function.conv2d(inp, wgt))


@pytest.mark.parametrize('num_bt, inp_ch, wgt_sz', [(1, 1, 3), (3, 16, 4), (4, 32, 5), (4, 4, 32)])
def test_forward(num_bt, inp_ch, wgt_sz, device):
    inp = torch.randn(4, num_bt, 64, 64, device=device)

    lyr0 = torch.nn.Conv2d(num_bt, inp_ch, wgt_sz, dtype=torch.float, device='cuda')
    lyr1 = trident.Conv2d(num_bt, inp_ch, wgt_sz)
    lyr1.weight, lyr1.bias = lyr0.weight, lyr0.bias

    assert util.equal(lyr0.forward(inp), lyr1.forward(inp))
