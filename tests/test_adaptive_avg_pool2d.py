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


@pytest.mark.parametrize('target_size', [2, 4, 8])
def test_function(target_size, dtype):
    inp = torch.randn(4, 4, 128, 128, dtype=dtype, device='cuda', requires_grad=True)
    assert util.equal(
        torch.nn.functional.adaptive_avg_pool2d(inp, target_size),
        trident.function.adaptive_avg_pool2d(inp, target_size)
    )


@pytest.mark.parametrize('target_size', [2, 4, 8])
def test_forward(target_size, dtype):
    inp = torch.randn(2, 256, 256, dtype=dtype, device='cuda', requires_grad=True)
    assert util.equal(
        torch.nn.AdaptiveAvgPool2d(target_size).forward(inp),
        trident.AdaptiveAvgPool2d(target_size).forward(inp)
    )
