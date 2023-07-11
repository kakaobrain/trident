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


def test_function(device):
    inp = torch.randn(512, 512, device=device)
    wgt = torch.randn(512, 512, device=device)

    assert util.equal(torch.nn.functional.linear(inp, wgt), trident.function.linear(inp, wgt))


def test_function_with_bias(device):
    inp = torch.randn(512, 512, device=device)
    wgt = torch.randn(512, 512, device=device)
    bis = torch.randn(512, device=device)

    assert util.equal(torch.nn.functional.linear(inp, wgt, bis), trident.function.linear(inp, wgt, bis))


@pytest.mark.parametrize("act", ["relu", "leaky_relu"])
def test_function_with_activation(act, device):
    inp = torch.randn(512, 512, device=device)
    wgt = torch.randn(512, 512, device=device)
    bis = torch.randn(512, device=device)

    assert util.equal(
        util.activate(torch.nn.functional.linear(inp, wgt, bis), act), trident.function.linear(inp, wgt, bis, act)
    )
