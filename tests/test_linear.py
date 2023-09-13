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


@pytest.mark.parametrize("num_batches, m_size, n_size, k_size", [(2, 512, 512, 100)])
def test_forward(num_batches, m_size, n_size, k_size, device):
    input = torch.randn(num_batches, m_size, k_size, device=device)
    weight = torch.randn(n_size, k_size, device=device)

    assert util.equal(torch.nn.functional.linear(input, weight), trident.function.linear(input, weight))

    bias = torch.randn(n_size, device=device)

    assert util.equal(torch.nn.functional.linear(input, weight, bias), trident.function.linear(input, weight, bias))

    input = input.permute(0, 2, 1)
    weight = weight.permute(1, 0)

    assert util.equal(torch.nn.functional.linear(input, weight), trident.function.linear(input, weight))


@pytest.mark.parametrize("num_batches, m_size, n_size, k_size", [(2, 512, 512, 100)])
def test_backward(num_batches, m_size, n_size, k_size, device):
    input = torch.randn(num_batches, m_size, k_size, device=device)
    weight = torch.randn(n_size, k_size, device=device)
    grad_output = torch.randn(num_batches, m_size, n_size, device=device)

    def train(func):
        i = input.clone()
        j = weight.clone()
        i.requires_grad = j.requires_grad = True
        func(i, j).backward(grad_output, retain_graph=True)
        return i.grad, j.grad

    (x, y) = train(torch.nn.functional.linear)
    (a, b) = train(trident.function.linear)

    assert util.equal(x, a)
    assert util.equal(y, b)

    input = input.permute(0, 2, 1).reshape(num_batches, m_size, k_size)
    weight = weight.permute(1, 0).reshape(n_size, k_size)

    (x, y) = train(torch.nn.functional.linear)
    (a, b) = train(trident.function.linear)

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

    (x, y, z) = train(torch.nn.functional.linear)
    (a, b, c) = train(trident.function.linear)

    assert util.equal(x, a)
    assert util.equal(y, b)
    assert util.equal(z, c)


@pytest.mark.parametrize("m_size, n_size, k_size", [(32, 32, 32)])
def test_linear(m_size, n_size, k_size, device, dtype):
    if dtype is torch.bfloat16:
        pytest.skip("Triton has a bug.")

    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(m_size, k_size, **factory_kwargs, requires_grad=True)
    weight = torch.randn(n_size, k_size, **factory_kwargs, requires_grad=True)
    bias = torch.randn(n_size, **factory_kwargs, requires_grad=True)
    grad_output = torch.randn(m_size, n_size, **factory_kwargs)

    operation = trident.Linear(m_size, n_size, **factory_kwargs)
    output = operation.forward(input)
    assert output is not None and output.dtype == dtype

    operation = trident.Linear(m_size, n_size, False, **factory_kwargs)
    output = operation.forward(input)
    assert output is not None and output.dtype == dtype

    trident.function.linear(input, weight, bias).backward(grad_output, retain_graph=True)

    assert input.grad is not None and input.grad.dtype == dtype
    assert weight.grad is not None and weight.grad.dtype == dtype
    assert bias.grad is not None and bias.grad.dtype == dtype
