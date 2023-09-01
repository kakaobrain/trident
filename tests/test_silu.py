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


@pytest.mark.parametrize("y_size, x_size", [(20, 10), (35, 80)])
def test_forward(y_size, x_size, device):
    input = torch.randn(y_size, x_size, device=device)

    assert util.equal(torch.nn.functional.silu(input), trident.function.silu(input))


@pytest.mark.parametrize("y_size, x_size", [(2, 64), (10, 90)])
def test_backward(y_size, x_size, device):
    input = torch.randn(y_size, x_size, device=device)
    grad_output = torch.rand_like(input)

    def train(func):
        i = input.clone()
        i.requires_grad = True
        func(i).backward(grad_output, retain_graph=True)
        return (i.grad,)

    (x,) = train(torch.nn.functional.silu)
    (a,) = train(trident.function.silu)

    assert util.equal(x, a)


@pytest.mark.parametrize("y_size, x_size", [(1, 32)])
def test_silu(y_size, x_size, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(y_size, x_size, **factory_kwargs)

    assert trident.SiLU().forward(input) is not None
