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


@pytest.mark.parametrize("num_vec, vec_sz", [(5, 32), (2, 30000)])
def test_function(num_vec, vec_sz, dtype, device):
    inp = torch.randn(num_vec, vec_sz, dtype=dtype, device=device)

    assert util.equal(
        torch.nn.functional.softmax(inp, 1), trident.function.softmax(inp, 1)
    )


@pytest.mark.parametrize("num_vec, vec_sz", [(3, 256), (2, 500)])
def test_forward(num_vec, vec_sz, dtype, device):
    inp = torch.randn(num_vec, vec_sz, dtype=dtype, device=device)

    assert util.equal(torch.nn.Softmax(1).forward(inp), trident.Softmax(1).forward(inp))


@pytest.mark.parametrize("num_vec, vec_sz", [(4, 64), (5, 70)])
def test_backward(num_vec, vec_sz, device):
    inp = torch.randn(num_vec, vec_sz, device=device)
    tgt = torch.randn(num_vec, vec_sz, device=device)

    x = inp.clone()
    a = inp.clone()
    x.requires_grad = a.requires_grad = True

    util.train(x, tgt, torch.nn.Softmax(1), torch.nn.CrossEntropyLoss())
    util.train(a, tgt, trident.Softmax(1), torch.nn.CrossEntropyLoss())

    assert util.equal(x.grad, a.grad)
