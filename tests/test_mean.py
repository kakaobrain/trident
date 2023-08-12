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


@pytest.mark.parametrize("y_size, x_size, dim", [(10000, 20000, 0), (10000, 20000, 1)])
def test_forward(y_size, x_size, dim, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(y_size, x_size, **factory_kwargs)

    assert util.equal(torch.mean(input, dim), trident.function.mean(input, dim))


@pytest.mark.parametrize("y_size, x_size, dim", [(10000, 20000, 0), (10000, 20000, 1)])
def test_backward(y_size, x_size, dim, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(y_size, x_size, **factory_kwargs)
    target = torch.randn(x_size if dim == 0 else y_size, **factory_kwargs)

    def train(func):
        i = input.clone()
        i.requires_grad = True
        func(i, dim).backward(target, retain_graph=True)
        return [i.grad]

    (x,) = train(torch.mean)
    (a,) = train(trident.function.mean)

    assert util.equal(x, a)


@pytest.mark.parametrize("y_size, x_size, dim", [(16, 32, 0), (16, 32, 1)])
def test_mean(y_size, x_size, dim, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(y_size, x_size, **factory_kwargs)

    operation = trident.Mean(dim)
    assert operation.forward(input) is not None
