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


@pytest.mark.parametrize("y_size, x_size", [(5, 32), (4, 64), (3, 128)])
def test_function(y_size, x_size, device):
    inp = torch.randn(y_size, x_size, device=device)

    assert util.equal(torch.nn.functional.relu(inp), trident.function.relu(inp))


@pytest.mark.parametrize("y_size, x_size", [(2, 256), (1, 512)])
def test_forward(y_size, x_size, device):
    inp = torch.randn(y_size, x_size, device=device)

    assert util.equal(torch.nn.ReLU().forward(inp), trident.ReLU().forward(inp))


@pytest.mark.parametrize("y_size, x_size", [(5, 32), (4, 64), (3, 128), (2, 256), (1, 512)])
def test_backward(y_size, x_size, device):
    inp = torch.randn(y_size, x_size, device=device)
    tgt = torch.randn(y_size, x_size, device=device)

    x = inp.clone()
    a = inp.clone()
    x.requires_grad = a.requires_grad = True

    util.train(x, tgt, torch.nn.ReLU())
    util.train(a, tgt, trident.ReLU())

    assert util.equal(x.grad, a.grad)


@pytest.mark.parametrize("y_size, x_size", [(1, 100)])
def test_relu(y_size, x_size, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(y_size, x_size, **factory_kwargs)

    output = trident.ReLU().forward(input)
    assert output is not None and output.dtype == dtype
