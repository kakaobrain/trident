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


@pytest.mark.parametrize("x_size, p", [(256, 0.0), (512, 1.0)])
def test_forward(x_size, p, device):
    input = torch.randn(x_size, device=device)

    assert util.equal(torch.nn.functional.dropout(input, p), trident.function.dropout(input, p))


@pytest.mark.parametrize("x_size, p", [(256, 0.0), (512, 1.0)])
def test_backward(x_size, p, device):
    input = torch.randn(x_size, device=device)
    grad_output = torch.rand_like(input)

    def train(func):
        i = input.clone()
        i.requires_grad = True
        func(i, p).backward(grad_output, retain_graph=True)
        return (i.grad,)

    (x,) = train(torch.nn.functional.dropout)
    (a,) = train(trident.function.dropout)

    assert util.equal(x, a)


@pytest.mark.parametrize("x_size, p", [(16, 0.7)])
def test_dropout(x_size, p, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(x_size, **factory_kwargs)

    assert trident.Dropout(p).forward(input) is not None
