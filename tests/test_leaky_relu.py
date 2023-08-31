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


@pytest.mark.parametrize("num_bt, num_elem", [(512, 512), (200, 300), (100, 1), (1, 100)])
def test_forward(num_bt, num_elem, device):
    inp = torch.randn(num_bt, num_elem, device=device)

    assert util.equal(torch.nn.functional.leaky_relu(inp), trident.function.leaky_relu(inp))


@pytest.mark.parametrize("num_bt, num_elem", [(512, 512), (200, 300), (100, 1), (1, 100)])
def test_backward(num_bt, num_elem, device):
    inp = torch.randn(num_bt, num_elem, device=device)
    tgt = torch.randn(num_bt, num_elem, device=device)

    x = inp.clone()
    a = inp.clone()
    x.requires_grad = a.requires_grad = True

    util.train(x, tgt, torch.nn.LeakyReLU())
    util.train(a, tgt, trident.LeakyReLU())

    assert util.equal(x.grad, a.grad)
