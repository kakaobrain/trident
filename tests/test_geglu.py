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


def geglu(input, weight, bias: torch.Tensor = None):
    state, gate = torch.nn.functional.linear(input, weight, bias).chunk(2, -1)
    return state * torch.nn.functional.gelu(gate)


# @pytest.mark.skip
@pytest.mark.parametrize("num_batches, m_size, n_size, k_size", [(2, 4, 4, 4)])
def test_forward(num_batches, m_size, n_size, k_size, device):
    input = torch.randn(m_size, k_size, device=device)
    weight = torch.randn(n_size, k_size, device=device)

    assert util.equal(geglu(input, weight), trident.function.geglu(input, weight))

    bias = torch.randn(n_size, device=device)

    assert util.equal(geglu(input, weight, bias), trident.function.geglu(input, weight, bias))


@pytest.mark.parametrize("num_batches, m_size, n_size, k_size", [(2, 4, 4, 4)])
def test_backward(num_batches, m_size, n_size, k_size, device):
    factory_kwargs = {"device": device}
    x_size = n_size // 2
    input = torch.randn(num_batches, m_size, k_size, **factory_kwargs)
    weight = torch.randn(n_size, k_size, **factory_kwargs)
    grad_output = torch.randn(num_batches, m_size, x_size, **factory_kwargs)

    def train(func):
        i = input.clone()
        j = weight.clone()
        i.requires_grad = j.requires_grad = True
        func(i, j).backward(grad_output, retain_graph=True)
        return i.grad, j.grad

    (x, y) = train(geglu)
    (a, b) = train(trident.function.geglu)

    assert util.equal(x, a)
    assert util.equal(y, b)

    bias = torch.randn(n_size, device=device)

    def train(func):
        i = input.clone()
        j = weight.clone()
        k = bias.clone()
        i.requires_grad = j.requires_grad = k.requires_grad = True
        func(i, j, k).backward(grad_output, retain_graph=True)
        return i.grad, j.grad, k.grad

    (x, y, z) = train(geglu)
    (a, b, c) = train(trident.function.geglu)

    assert util.equal(x, a)
    assert util.equal(y, b)
    assert util.equal(z, c)


@pytest.mark.parametrize("num_batches, m_size, n_size, k_size", [(2, 64, 64, 128)])
def test_geglu(num_batches, m_size, n_size, k_size, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(num_batches, m_size, k_size, **factory_kwargs)

    output = trident.GEGLU(m_size, n_size, **factory_kwargs).forward(input)

    assert output is not None
    assert output.dtype == dtype

    output = trident.GEGLU(m_size, n_size, False, **factory_kwargs).forward(input)

    assert output is not None
    assert output.dtype == dtype
