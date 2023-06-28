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


@pytest.mark.parametrize('knl_sz', [2, 3, 4, 5, 6, 7, 8])
def test_function(knl_sz, dtype, device):
    inp = torch.randn(2, 3, 128, 128, dtype=dtype, device=device)

    assert util.equal(torch.nn.functional.max_pool2d(inp, knl_sz), trident.function.max_pool2d(inp, knl_sz))


@pytest.mark.parametrize('knl_sz', [32, 64, 96, 128])
def test_forward(knl_sz, dtype, device):
    inp = torch.randn(10, 7, 256, 256, dtype=dtype, device=device)

    assert util.equal(torch.nn.MaxPool2d(knl_sz).forward(inp), trident.MaxPool2d(knl_sz).forward(inp))
