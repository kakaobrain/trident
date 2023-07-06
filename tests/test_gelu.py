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


@pytest.mark.parametrize('vec_sz', [10, 16])
def test_function(vec_sz, dtype, device):
    inp = torch.randn(vec_sz, dtype=dtype, device=device)

    assert util.equal(torch.nn.functional.gelu(inp), trident.function.gelu(inp))


@pytest.mark.parametrize('vec_sz', [3, 16384])
def test_forward(vec_sz, dtype, device):
    inp = torch.randn(vec_sz, dtype=dtype, device=device)

    assert util.equal(torch.nn.GELU().forward(inp), trident.GELU().forward(inp))


@pytest.mark.parametrize('vec_sz', [1, 20000])
def test_backward(vec_sz, dtype, device):
    inp = torch.randn(vec_sz, dtype=dtype, device=device)
    tgt = torch.randn(vec_sz, dtype=dtype, device=device)

    x = inp.clone()
    a = inp.clone()
    x.requires_grad = a.requires_grad = True

    lyr0 = torch.nn.GELU()
    lyr1 = trident.GELU()

    util.train(x, tgt, lyr0)
    util.train(a, tgt, lyr1)

    assert util.equal(x.grad, a.grad)
