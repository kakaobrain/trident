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


@pytest.mark.parametrize("y_size, x_size", [(512, 512), (200, 300), (100, 1), (1, 100)])
def test_forward(y_size, x_size, device):
    input = torch.randn(y_size, x_size, device=device)
    weight = torch.randn(x_size, device=device)

    assert util.equal(torch.nn.functional.prelu(input, weight), trident.function.prelu(input, weight))


@pytest.mark.parametrize("y_size, x_size", [(512, 512), (200, 300), (100, 1), (1, 100)])
def test_backward(y_size, x_size, device):
    input = torch.randn(y_size, x_size, device=device)
    target = torch.randn(y_size, x_size, device=device)

    x = input.clone()
    a = input.clone()
    x.requires_grad = a.requires_grad = True

    y = torch.nn.PReLU(x_size, 0.3, device=device)
    b = trident.PReLU(x_size, 0.3, device=device)

    util.train(x, target, y)
    util.train(a, target, b)

    assert util.equal(x.grad, a.grad)
    assert util.equal(y.weight.grad, b.weight.grad)
