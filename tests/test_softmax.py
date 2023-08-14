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


@pytest.mark.parametrize("y_size, x_size, dim", [(5, 32, 1), (2, 30000, 1)])
def test_forward(y_size, x_size, dim, dtype, device):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(y_size, x_size, **factory_kwargs)

    assert util.equal(torch.nn.functional.softmax(input, dim), trident.function.softmax(input, dim))


@pytest.mark.parametrize("num_vec, vec_sz", [(4, 64), (5, 70)])
def test_backward(num_vec, vec_sz, device, dtype):
    ctor_args = {"device": device, "dtype": dtype}
    inp = torch.randn(num_vec, vec_sz, **ctor_args)
    tgt = torch.randn(num_vec, vec_sz, **ctor_args)

    def train(func, dim):
        i = inp.clone()
        i.requires_grad = True
        func(i, dim).backward(tgt, retain_graph=True)
        return [i.grad]

    (x,) = train(torch.nn.functional.softmax, 1)
    (a,) = train(trident.function.softmax, 1)

    assert util.equal(x, a)


@pytest.mark.parametrize("y_size, x_size, dim", [(1, 32, 1)])
def test_softmax(y_size, x_size, dim, dtype, device):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(y_size, x_size, **factory_kwargs)

    assert trident.Softmax(dim).forward(input) is not None
