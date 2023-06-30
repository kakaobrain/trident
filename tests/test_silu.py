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


@pytest.mark.parametrize("num_vec, vec_sz", [(1, 32), (3, 40)])
def test_function(num_vec, vec_sz, dtype, device):
    inp = torch.randn(num_vec, vec_sz, dtype=dtype, device=device)

    assert util.equal(torch.nn.functional.silu(inp), trident.function.silu(inp))


@pytest.mark.parametrize("num_vec, vec_sz", [(20, 10), (35, 80)])
def test_forward(num_vec, vec_sz, dtype, device):
    inp = torch.randn(num_vec, vec_sz, dtype=dtype, device=device)

    assert util.equal(torch.nn.SiLU().forward(inp), trident.SiLU().forward(inp))


@pytest.mark.parametrize("num_vec, vec_sz", [(1, 64), (10, 90)])
def test_backward(num_vec, vec_sz, dtype, device):
    inp = torch.randn(num_vec, vec_sz, dtype=dtype, device=device)
    tgt = torch.randn(num_vec, vec_sz, dtype=dtype, device=device)

    x = inp.clone()
    a = inp.clone()
    x.requires_grad = a.requires_grad = True

    util.train(x, tgt, torch.nn.SiLU())
    util.train(a, tgt, trident.SiLU())

    assert util.equal(x.grad, a.grad)
