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


@pytest.mark.parametrize("y_size, x_size, dim", [(4, 2000, 0), (2000, 4, 1)])
def test_forward(y_size, x_size, dim, device):
    input = torch.randn(y_size, x_size, device=device)

    assert util.equal(torch.var(input, dim), trident.function.var(input, dim))


@pytest.mark.parametrize("y_size, x_size, dim", [(2000, 4, 0), (4, 2000, 1)])
def test_backward(y_size, x_size, dim, device):
    input = torch.randn(y_size, x_size, device=device)
    target = torch.randn(x_size if dim == 0 else y_size, device=device)

    def train(func):
        i = input.clone()
        i.requires_grad = True
        func(i, dim).backward(target, retain_graph=True)
        return (i.grad,)

    (x,) = train(torch.var)
    (a,) = train(trident.function.var)

    assert util.equal(x, a)


@pytest.mark.parametrize("y_size, x_size, dim", [(1, 16, 0), (16, 1, 1)])
def test_mean(y_size, x_size, dim, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(y_size, x_size, **factory_kwargs)

    assert trident.Var(dim).forward(input) is not None
