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
    "num_bt, inp_feat, out_feat, act",
    [(512, 512, 100, "relu"), (200, 120, 15, "leaky_relu")],
)
def test_forward(num_bt, inp_feat, out_feat, act, device):
    inp = torch.randn(num_bt, inp_feat, device=device)
    wgt = torch.randn(out_feat, inp_feat, device=device)
    bis = torch.randn(out_feat, device=device)

    assert util.equal(
        torch.nn.functional.linear(inp, wgt), trident.function.linear(inp, wgt)
    )

    assert util.equal(
        torch.nn.functional.linear(inp, wgt, bis),
        trident.function.linear(inp, wgt, bis),
    )

    assert util.equal(
        util.activate(torch.nn.functional.linear(inp, wgt, bis), act),
        trident.function.linear(inp, wgt, bis, act),
    )


@pytest.mark.parametrize(
    "num_bt, inp_feat, out_feat, act",
    [(12, 34, 3, "relu"), (43, 12, 2, "leaky_relu")],
)
def test_backward(num_bt, inp_feat, out_feat, act, device):
    inp = torch.randn(num_bt, inp_feat, device=device)
    wgt = torch.randn(out_feat, inp_feat, device=device)
    bis = torch.randn(out_feat, device=device)

    tgt = torch.randn(num_bt, out_feat, device=device)

    x = inp.clone()
    a = inp.clone()
    x.requires_grad = a.requires_grad = True

    lyr0 = torch.nn.Linear(inp_feat, out_feat, True, device=device)
    lyr0.weight = torch.nn.Parameter(wgt)
    lyr0.bias = torch.nn.Parameter(bis)

    lyr1 = trident.Linear(inp_feat, out_feat, True, activation=act)
    lyr1.weight = torch.nn.Parameter(wgt)
    lyr1.bias = torch.nn.Parameter(bis)

    util.train(x, tgt, lyr0, act)
    util.train(a, tgt, lyr1)

    assert util.equal(x.grad, a.grad)
    assert util.equal(lyr0.weight.grad, lyr1.weight.grad)
    assert util.equal(lyr0.bias.grad, lyr1.bias.grad)
