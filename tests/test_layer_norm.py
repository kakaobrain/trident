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


@pytest.mark.parametrize('num_vec, vec_sz', [(3, 16), (1, 20000)])
def test_forward(num_vec, vec_sz, dtype, device):
    inp = torch.randn(num_vec, vec_sz, dtype=dtype, device=device)
    norm_sh = [inp.shape[-1], ]

    assert util.equal(torch.nn.functional.layer_norm(inp, norm_sh), trident.function.layer_norm(inp, norm_sh))

    wgt = torch.randn(norm_sh, dtype=dtype, device=device)

    assert util.equal(
        torch.nn.functional.layer_norm(inp, norm_sh, wgt, None), trident.function.layer_norm(inp, norm_sh, wgt, None)
    )

    bis = torch.randn(norm_sh, dtype=dtype, device=device)

    assert util.equal(
        torch.nn.functional.layer_norm(inp, norm_sh, None, bis), trident.function.layer_norm(inp, norm_sh, None, bis)
    )
    assert util.equal(
        torch.nn.functional.layer_norm(inp, norm_sh, wgt, bis), trident.function.layer_norm(inp, norm_sh, wgt, bis)
    )


@pytest.mark.parametrize('num_vec, vec_sz, elem_afn', [(3, 10, False), (11, 40, True)])
def test_backward(num_vec, vec_sz, elem_afn, dtype, device):
    inp = torch.randn(num_vec, vec_sz, dtype=dtype, device=device)
    tgt = torch.randn(num_vec, vec_sz, dtype=dtype, device=device)
    norm_sh = [inp.shape[-1]]

    x = inp.clone()
    a = inp.clone()
    x.requires_grad = a.requires_grad = True

    lyr0 = torch.nn.LayerNorm(norm_sh, elementwise_affine=elem_afn, dtype=dtype, device=device)
    lyr1 = trident.LayerNorm(norm_sh, elementwise_affine=elem_afn, dtype=dtype, device=device)

    util.train(x, tgt, lyr0)
    util.train(a, tgt, lyr1)

    assert util.equal(x.grad, a.grad)

    if elem_afn:
        assert util.equal(lyr0.weight.grad, lyr1.weight.grad)
        assert util.equal(lyr0.bias.grad, lyr1.bias.grad)
