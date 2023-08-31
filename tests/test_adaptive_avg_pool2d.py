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


@pytest.mark.parametrize("tgt_sz", [2, 4, 8])
def test_function(tgt_sz, device):
    inp = torch.randn(4, 4, 128, 128, device=device)

    assert util.equal(
        torch.nn.functional.adaptive_avg_pool2d(inp, tgt_sz), trident.function.adaptive_avg_pool2d(inp, tgt_sz)
    )


@pytest.mark.parametrize("tgt_sz", [2, 4, 8])
def test_forward(tgt_sz, device):
    inp = torch.randn(2, 256, 256, device=device)

    assert util.equal(torch.nn.AdaptiveAvgPool2d(tgt_sz).forward(inp), trident.AdaptiveAvgPool2d(tgt_sz).forward(inp))
