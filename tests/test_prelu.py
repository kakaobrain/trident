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


@pytest.mark.parametrize("y_size, x_size", [(128, 512), (200, 300)])
def test_forward(y_size, x_size, device):
    input = torch.randn(y_size, x_size, device=device)
    weight = torch.randn(x_size, device=device)

    assert util.equal(torch.nn.functional.prelu(input, weight), trident.function.prelu(input, weight))


@pytest.mark.parametrize("y_size, x_size", [(512, 128), (100, 600)])
def test_backward(y_size, x_size, device):
    input = torch.randn(y_size, x_size, device=device)
    weight = torch.randn(x_size, device=device)
    grad_output = torch.rand_like(input)

    def train(func):
        i = input.clone()
        j = weight.clone()
        i.requires_grad = j.requires_grad = True
        func(i, j).backward(grad_output, retain_graph=True)
        return i.grad, j.grad

    (x, y) = train(torch.nn.functional.prelu)
    (a, b) = train(trident.function.prelu)

    assert util.equal(x, a)
    assert util.equal(y, b)


@pytest.mark.parametrize("y_size, x_size", [(1, 100)])
def test_prelu(y_size, x_size, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(y_size, x_size, **factory_kwargs)

    output = trident.PReLU(x_size, 0.3, **factory_kwargs).forward(input)

    assert output is not None
    assert output.dtype == dtype
