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


@pytest.mark.parametrize("y_size, x_size", [(2, 256), (3, 200)])
def test_forward(y_size, x_size, device):
    input = torch.randn(y_size, x_size, device=device)

    assert util.equal(torch.nn.functional.relu(input), trident.function.relu(input))


@pytest.mark.parametrize("y_size, x_size", [(2, 256), (3, 200)])
def test_backward(y_size, x_size, device):
    input = torch.randn((y_size, x_size), device=device)
    grad_output = torch.randn((y_size, x_size), device=device)

    def train(func):
        i = input.clone()
        i.requires_grad = True
        func(i).backward(grad_output, retain_graph=True)
        return (i.grad,)

    (x,) = train(torch.nn.functional.relu)
    (a,) = train(trident.function.relu)

    assert util.equal(x, a)


@pytest.mark.parametrize("y_size, x_size", [(1, 100)])
def test_relu(y_size, x_size, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(y_size, x_size, **factory_kwargs, requires_grad=True)
    grad_output = torch.randn_like(input)

    output = trident.ReLU().forward(input)

    assert output is not None
    assert output.dtype == dtype

    output.backward(grad_output)
    assert input.grad is not None
    assert input.grad.dtype == dtype
