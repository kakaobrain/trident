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


def masked_softmax(input: torch.Tensor, mask: torch.Tensor, dim: int):
    input = torch.where(mask.bool(), float("-inf"), input)
    output = torch.nn.functional.softmax(input, dim)

    return output


def build_mask(y_size: int, x_size: int, device=None):
    mask = torch.randint(0, 2, (y_size, x_size), device=device)
    mask[0, :] = mask[:, 0] = 0

    return mask


@pytest.mark.parametrize("y_size, x_size, dim", [(2, 512, 0), (3, 1000, 1)])
def test_forward(y_size, x_size, dim, device):
    input = torch.randn(y_size, x_size, device=device)
    mask = build_mask(y_size, x_size, device)

    assert util.equal(masked_softmax(input, mask, dim), trident.function.masked_softmax(input, mask, dim))


@pytest.mark.parametrize("y_size, x_size, dim", [(3, 1000, 0), (2, 512, 1)])
def test_backward(y_size, x_size, dim, device):
    input = torch.randn(y_size, x_size, device=device)
    mask = build_mask(y_size, x_size, device)
    grad_output = torch.randn(y_size, x_size, device=device)

    def train(func):
        i = input.clone()
        i.requires_grad = True
        func(i, mask, dim).backward(grad_output, retain_graph=True)
        return (i.grad,)

    (x,) = train(masked_softmax)
    (a,) = train(trident.function.masked_softmax)

    assert util.equal(x, a)


@pytest.mark.parametrize("y_size, x_size, dim", [(1, 32, 1)])
def test_masked_softmax(y_size, x_size, dim, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(y_size, x_size, **factory_kwargs, requires_grad=True)
    mask = build_mask(y_size, x_size, device)
    grad_output = torch.randn_like(input)

    output = trident.MaskedSoftmax(dim).forward(input, mask)

    assert output is not None
    assert output.dtype == dtype

    output.backward(grad_output)

    assert input.grad is not None
    assert input.grad.dtype == dtype
