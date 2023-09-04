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


@pytest.mark.parametrize("y_size, x_size, dim", [(10, 2000, 0), (2000, 10, 1)])
def test_forward(y_size, x_size, dim, device):
    input = torch.randn(y_size, x_size, device=device)

    (x, y) = torch.max(input, dim)
    (i, j) = trident.function.max(input, dim)

    assert util.equal(x, i)
    assert util.equal(y, j)


@pytest.mark.parametrize("y_size, x_size, dim", [(2000, 10, 0), (10, 2000, 1)])
def test_backward(y_size, x_size, dim, device):
    input = torch.randn(y_size, x_size, device=device)
    target = torch.randn(x_size if dim == 0 else y_size, device=device)

    def train(func):
        i = input.clone()
        i.requires_grad = True
        j, k = func(i, dim)
        j.backward(target, retain_graph=True)
        return [i.grad]

    (x,) = train(torch.max)
    (a,) = train(trident.function.max)

    assert util.equal(x, a)


@pytest.mark.parametrize("y_size, x_size, dim", [(16, 32, 0), (16, 32, 1)])
def test_max(y_size, x_size, dim, device, dtype):
    if dtype == torch.bfloat16:
        pytest.skip("Skipping due to bfloat16 dtype")

    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(y_size, x_size, **factory_kwargs)

    operation = trident.Max(dim)
    assert operation.forward(input) is not None
