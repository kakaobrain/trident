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


@pytest.mark.parametrize("y_size, x_size", [(512, 512), (200, 300), (100, 1), (1, 100)])
def test_forward(y_size, x_size, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}

    input = torch.randn(y_size, x_size, **factory_kwargs)
    weight = torch.randn(x_size, **factory_kwargs)

    assert util.equal(
        torch.nn.functional.prelu(input, weight), trident.function.prelu(input, weight)
    )


@pytest.mark.parametrize("y_size, x_size", [(512, 512), (200, 300), (100, 1), (1, 100)])
def test_backward(y_size, x_size, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}

    input = torch.randn(y_size, x_size, **factory_kwargs)
    target = torch.randn(y_size, x_size, **factory_kwargs)

    x = input.clone()
    a = input.clone()
    x.requires_grad = a.requires_grad = True

    y = torch.nn.PReLU(x_size, 0.3, **factory_kwargs)
    b = trident.PReLU(x_size, 0.3, **factory_kwargs)

    util.train(x, target, y)
    util.train(a, target, b)

    assert util.equal(x.grad, a.grad)
    assert util.equal(y.weight.grad, b.weight.grad)
