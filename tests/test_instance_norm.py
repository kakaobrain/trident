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


@pytest.mark.parametrize("num_ch, w, h", [(1, 64, 64), (5, 20, 20)])
def test_function(num_ch, w, h, dtype, device):
    inp = torch.randn(num_ch, w, h, dtype=dtype, device=device)

    assert util.equal(
        torch.nn.functional.instance_norm(inp),
        trident.function.instance_norm(inp),
    )


@pytest.mark.parametrize("num_bt, num_ch, w, h", [(1, 3, 256, 256), (4, 4, 150, 150)])
def test_forward(num_bt, num_ch, w, h, dtype, device):
    inp = torch.randn(num_bt, num_ch, w, h, dtype=dtype, device=device)

    assert util.equal(
        torch.nn.InstanceNorm2d(num_ch, dtype=dtype, device=device).forward(inp),
        trident.InstanceNorm2d(num_ch, dtype=dtype, device=device).forward(inp),
    )
