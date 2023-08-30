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


@pytest.mark.parametrize("y_size, x_size", [(2000, 4), (4, 2000)])
def test_forward(y_size, x_size, device):
    factory_kwargs = {"device": device}
    input = torch.randn(y_size, x_size, **factory_kwargs)
    normalized_shape = (input.shape[-1],)

    assert util.equal(
        torch.nn.functional.layer_norm(input, normalized_shape),
        trident.function.layer_norm(input, normalized_shape),
    )

    weight = torch.randn(normalized_shape, **factory_kwargs)

    assert util.equal(
        torch.nn.functional.layer_norm(input, normalized_shape, weight),
        trident.function.layer_norm(input, normalized_shape, weight),
    )

    bias = torch.randn(normalized_shape, **factory_kwargs)

    assert util.equal(
        torch.nn.functional.layer_norm(input, normalized_shape, None, bias),
        trident.function.layer_norm(input, normalized_shape, None, bias),
    )
    assert util.equal(
        torch.nn.functional.layer_norm(input, normalized_shape, weight, bias),
        trident.function.layer_norm(input, normalized_shape, weight, bias),
    )


@pytest.mark.parametrize("y_size, x_size", [(2000, 4), (4, 2000)])
def test_backward(y_size, x_size, device):
    factory_kwargs = {"device": device}
    input = torch.randn(y_size, x_size, **factory_kwargs)
    grad_output = torch.randn(y_size, x_size, **factory_kwargs)
    normalized_shape = (input.shape[-1],)

    def train(func):
        i = input.clone()
        i.requires_grad = True
        func(i, normalized_shape).backward(grad_output, retain_graph=True)
        return (i.grad,)

    (x,) = train(torch.layer_norm)
    (a,) = train(trident.function.layer_norm)

    assert util.equal(x, a)

    weight = torch.randn(normalized_shape, **factory_kwargs)

    def train(func):
        i = input.clone()
        j = weight.clone()
        i.requires_grad = j.requires_grad = True
        func(i, normalized_shape, j).backward(grad_output, retain_graph=True)
        return i.grad, j.grad

    (x, y) = train(torch.layer_norm)
    (a, b) = train(trident.function.layer_norm)

    assert util.equal(x, a)
    assert util.equal(y, b)

    bias = torch.randn(normalized_shape, **factory_kwargs)

    def train(func):
        i = input.clone()
        j = bias.clone()
        i.requires_grad = j.requires_grad = True
        func(i, normalized_shape, None, j).backward(grad_output, retain_graph=True)
        return i.grad, j.grad

    (x, y) = train(torch.layer_norm)
    (a, b) = train(trident.function.layer_norm)

    assert util.equal(x, a)
    assert util.equal(y, b)

    def train(func):
        i = input.clone()
        j = weight.clone()
        k = bias.clone()
        i.requires_grad = j.requires_grad = k.requires_grad = True
        func(i, normalized_shape, j, k).backward(grad_output, retain_graph=True)
        return i.grad, j.grad, k.grad

    (x, y, z) = train(torch.layer_norm)
    (a, b, c) = train(trident.function.layer_norm)

    assert util.equal(x, a)
    assert util.equal(y, b)
    assert util.equal(z, c)
