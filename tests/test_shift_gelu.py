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


def shift_gelu(input: torch.Tensor, bias: torch.Tensor):
    return torch.nn.functional.gelu(input + bias)


@pytest.mark.parametrize("num_batches, y_size, x_size", [(2, 10, 1000)])
def test_forward(num_batches, y_size, x_size, device):
    input = torch.randn((num_batches, y_size, x_size), device=device)
    bias = torch.randn(x_size, device=device)

    assert util.equal(trident.function.shift_gelu(input, bias), shift_gelu(input, bias))


@pytest.mark.parametrize("num_batches, y_size, x_size", [(2, 10, 1000)])
def test_backward(num_batches, y_size, x_size, device):
    input = torch.randn((num_batches, y_size, x_size), device=device)
    bias = torch.randn(x_size, device=device)
    grad_output = torch.rand_like(input)

    def train(func):
        i = input.clone()
        j = bias.clone()
        i.requires_grad = j.requires_grad = True
        func(i, j).backward(grad_output, retain_graph=True)
        return i.grad, j.grad

    (x, y) = train(trident.function.shift_gelu)
    (a, b) = train(shift_gelu)

    assert util.equal(x, a)
    assert util.equal(y, b)


@pytest.mark.parametrize("num_batches, y_size, x_size", [(2, 10, 1000)])
def test_shift_gelu(num_batches, y_size, x_size, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn((num_batches, y_size, x_size), **factory_kwargs)

    assert trident.ShiftGELU(x_size, **factory_kwargs).forward(input) is not None
