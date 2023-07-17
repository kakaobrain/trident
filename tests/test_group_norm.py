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


@pytest.mark.parametrize(
    "num_vec, vec_sz, num_grp, afn", [(3, 16, 2, False), (7, 60000, 2, True)]
)
def test_forward(num_vec, vec_sz, num_grp, afn, dtype, device):
    inp = torch.randn(num_vec, vec_sz, dtype=dtype, device=device)
    wgt = torch.randn(vec_sz, dtype=dtype, device=device)
    bis = torch.randn(vec_sz, dtype=dtype, device=device)

    assert util.equal(
        torch.nn.functional.group_norm(inp, num_grp),
        trident.function.group_norm(inp, num_grp),
    )
    assert util.equal(
        torch.nn.functional.group_norm(inp, num_grp, wgt, bis),
        trident.function.group_norm(inp, num_grp, wgt, bis),
    )


@pytest.mark.parametrize(
    "num_vec, vec_sz, num_grp, afn", [(2, 4, 2, True), (3, 60000, 3, True)]
)
def test_backward(num_vec, vec_sz, num_grp, afn, dtype, device):
    inp = torch.randn(num_vec, vec_sz, dtype=dtype, device=device)
    tgt = torch.randn(num_vec, vec_sz, dtype=dtype, device=device)

    a = inp.clone()
    b = inp.clone()
    a.requires_grad = b.requires_grad = True

    lyr0 = torch.nn.GroupNorm(num_grp, vec_sz, affine=afn, dtype=dtype, device=device)
    lyr1 = trident.GroupNorm(num_grp, vec_sz, affine=afn, dtype=dtype, device=device)
    util.train(a, tgt, lyr0)
    util.train(b, tgt, lyr1)
    assert util.equal(a.grad, b.grad)

    if afn:
        assert util.equal(lyr0.weight.grad, lyr1.weight.grad)
        assert util.equal(lyr0.bias.grad, lyr1.bias.grad)
