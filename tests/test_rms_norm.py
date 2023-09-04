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


def rms_norm(input: torch.Tensor, p: float, weight: torch.Tensor, bias: torch.Tensor = None, eps: float = 1e-08):
    y_size, x_size = input.shape

    if p < 0.0 or p > 1.0:
        norm = input.norm(2, dim=-1, keepdim=True)
        partial_size = x_size
    else:
        partial_size = int(x_size * p)
        partial_input, _ = torch.split(input, [partial_size, x_size - partial_size], dim=-1)
        norm = partial_input.norm(2, dim=-1, keepdim=True)

    rms = norm * partial_size ** (-1.0 / 2)
    output = input / (rms + eps)

    if bias is not None:
        return weight * output + bias

    return weight * output


@pytest.mark.parametrize("y_size, x_size, p", [(1000, 10, 0.5), (10, 1000, 1.0)])
def test_forward(y_size, x_size, p, device):
    input = torch.randn(y_size, x_size, device=device)
    weight = torch.randn(x_size, device=device)

    assert util.equal(rms_norm(input, p, weight), trident.function.rms_norm(input, p, weight))

    bias = torch.randn(x_size, device=device)

    assert util.equal(rms_norm(input, p, weight, bias), trident.function.rms_norm(input, p, weight, bias))


@pytest.mark.parametrize("y_size, x_size, p", [(1000, 10, 0.5), (10, 1000, 1.0)])
def test_backward(y_size, x_size, p, device):
    input = torch.randn((y_size, x_size), device=device)
    weight = torch.randn(x_size, device=device)
    grad_output = torch.randn(y_size, x_size, device=device)

    def train(func):
        i = input.clone()
        j = weight.clone()
        i.requires_grad = j.requires_grad = True
        func(i, p, j).backward(grad_output, retain_graph=True)
        return i.grad, j.grad

    (x, y) = train(rms_norm)
    (a, b) = train(trident.function.rms_norm)

    assert util.equal(x, a)
    assert util.equal(y, b)

    bias = torch.randn(x_size, device=device)

    def train(func):
        i = input.clone()
        j = weight.clone()
        k = bias.clone()
        i.requires_grad = j.requires_grad = k.requires_grad = True
        func(i, p, j, k).backward(grad_output, retain_graph=True)
        return i.grad, j.grad, k.grad

    (x, y, z) = train(rms_norm)
    (a, b, c) = train(trident.function.rms_norm)

    assert util.equal(x, a)
    assert util.equal(y, b)
    assert util.equal(z, c)


@pytest.mark.parametrize("y_size, x_size", [(4, 16)])
def test_rms_norm(y_size, x_size, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn((y_size, x_size), **factory_kwargs)

    output = trident.RMSNorm(x_size, **factory_kwargs).forward(input)
    assert output is not None and output.dtype == dtype
    output = trident.RMSNorm(x_size, 0.5, **factory_kwargs).forward(input)
    assert output is not None and output.dtype == dtype
    output = trident.RMSNorm(x_size, bias=True, **factory_kwargs).forward(input)
    assert output is not None and output.dtype == dtype
    output = trident.RMSNorm(x_size, 0.5, bias=True, **factory_kwargs).forward(input)
    assert output is not None and output.dtype == dtype
