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


@pytest.mark.parametrize("m_size, n_size, k_size", [(512, 512, 100), (200, 120, 15)])
def test_forward(m_size, n_size, k_size, device):
    factory_kwargs = {"device": device}
    input = torch.randn(m_size, k_size, **factory_kwargs)
    weight = torch.randn(n_size, k_size, **factory_kwargs)

    assert util.equal(torch.nn.functional.linear(input, weight), trident.function.linear(input, weight))

    bias = torch.randn(n_size, **factory_kwargs)

    assert util.equal(torch.nn.functional.linear(input, weight, bias), trident.function.linear(input, weight, bias))


@pytest.mark.parametrize("m_size, n_size, k_size", [(256, 512, 32), (200, 120, 15)])
def test_backward(m_size, n_size, k_size, device):
    factory_kwargs = {"device": device}
    input = torch.randn(m_size, k_size, **factory_kwargs)
    weight = torch.randn(n_size, k_size, **factory_kwargs)
    target = torch.randn(m_size, n_size, **factory_kwargs)

    def train(func):
        i = input.clone()
        j = weight.clone()
        i.requires_grad = j.requires_grad = True
        func(i, j).backward(target, retain_graph=True)
        return (
            i.grad,
            j.grad,
        )

    (x, y) = train(torch.nn.functional.linear)
    (a, b) = train(trident.function.linear)

    assert util.equal(x, a)
    assert util.equal(y, b)

    bias = torch.randn(n_size, **factory_kwargs)

    def train(func):
        i = input.clone()
        j = weight.clone()
        k = bias.clone()
        i.requires_grad = j.requires_grad = k.requires_grad = True
        func(i, j, k).backward(target, retain_graph=True)
        return (
            i.grad,
            j.grad,
            k.grad,
        )

    (x, y, z) = train(torch.nn.functional.linear)
    (a, b, c) = train(trident.function.linear)

    assert util.equal(x, a)
    assert util.equal(y, b)
    assert util.equal(z, c)


@pytest.mark.parametrize("m_size, n_size, k_size", [(32, 32, 32)])
def test_linear(m_size, n_size, k_size, device):
    factory_kwargs = {"device": device}
    input = torch.randn(m_size, k_size, **factory_kwargs)

    operation = trident.Linear(m_size, n_size, **factory_kwargs)
    assert operation.forward(input) is not None

    operation = trident.Linear(m_size, n_size, False, **factory_kwargs)
    assert operation.forward(input) is not None
