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


def test_function(dtype, device):
    inp = torch.randn(8, dtype=dtype, device=device)
    p = 0.0

    assert util.equal(torch.nn.functional.dropout(inp, p), trident.function.dropout(inp, p))


def test_forward(dtype, device):
    inp = torch.randn(16, dtype=dtype, device=device)
    p = 1.0

    assert util.equal(torch.nn.Dropout(p).forward(inp), trident.Dropout(p).forward(inp))


@pytest.mark.parametrize('p', [0.0, 1.0])
def test_backward(p, dtype, device):
    inp = torch.randn(8, dtype=dtype, device=device)
    tgt = torch.randn(8, dtype=dtype, device=device)

    x = inp.clone()
    a = inp.clone()
    x.requires_grad = a.requires_grad = True

    util.train(x, tgt, torch.nn.Dropout(p))
    util.train(a, tgt, trident.Dropout(p))

    assert util.equal(x.grad, a.grad)
