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


@pytest.mark.parametrize("num_batches, y_size, x_size, num_groups", [(2, 16, 10, 4), (16, 1000, 40, 1000)])
def test_forward(num_batches, y_size, x_size, num_groups, device):
    input = torch.randn((num_batches, y_size, x_size), device=device)

    assert util.equal(torch.nn.functional.group_norm(input, num_groups), trident.function.group_norm(input, num_groups))

    weight = torch.randn(y_size, device=device)

    assert util.equal(
        torch.nn.functional.group_norm(input, num_groups, weight),
        trident.function.group_norm(input, num_groups, weight),
    )

    bias = torch.rand_like(weight)

    assert util.equal(
        torch.nn.functional.group_norm(input, num_groups, None, bias),
        trident.function.group_norm(input, num_groups, None, bias),
    )
    assert util.equal(
        torch.nn.functional.group_norm(input, num_groups, weight, bias),
        trident.function.group_norm(input, num_groups, weight, bias),
    )


@pytest.mark.parametrize("num_batches, y_size, x_size, num_groups", [(2, 16, 10, 4), (16, 1000, 40, 1000)])
def test_backward(num_batches, y_size, x_size, num_groups, device):
    input = torch.randn((num_batches, y_size, x_size), device=device)
    grad_output = torch.rand_like(input)

    def train(func):
        i = input.clone()
        i.requires_grad = True
        func(i, num_groups).backward(grad_output, retain_graph=True)
        return (i.grad,)

    (x,) = train(torch.group_norm)
    (a,) = train(trident.function.group_norm)

    assert util.equal(x, a)

    weight = torch.randn(y_size, device=device)

    def train(func):
        i = input.clone()
        j = weight.clone()
        i.requires_grad = j.requires_grad = True
        func(i, num_groups, j).backward(grad_output, retain_graph=True)
        return i.grad, j.grad

    (x, y) = train(torch.group_norm)
    (a, b) = train(trident.function.group_norm)

    assert util.equal(x, a)
    assert util.equal(y, b)

    bias = torch.randn(y_size, device=device)

    def train(func):
        i = input.clone()
        j = weight.clone()
        k = bias.clone()
        i.requires_grad = j.requires_grad = k.requires_grad = True
        func(i, num_groups, j, k).backward(grad_output, retain_graph=True)
        return i.grad, j.grad, k.grad

    (x, y, z) = train(torch.group_norm)
    (a, b, c) = train(trident.function.group_norm)

    assert util.equal(x, a)
    assert util.equal(y, b)
    assert util.equal(z, c)


@pytest.mark.parametrize("num_batches, y_size, x_size, num_groups", [(1, 8, 1, 4)])
def test_group_norm(num_batches, y_size, x_size, num_groups, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(num_batches, y_size, x_size, **factory_kwargs, requires_grad=True)
    grad_output = torch.rand_like(input)

    operation = trident.GroupNorm(num_groups, y_size, **factory_kwargs)
    output = operation.forward(input)
    assert output is not None and output.dtype == dtype

    operation = trident.GroupNorm(num_groups, y_size, affine=True, **factory_kwargs)
    output = operation.forward(input)
    assert output is not None and output.dtype == dtype

    output.backward(grad_output)
    assert input.grad is not None
    assert input.grad.dtype == dtype
