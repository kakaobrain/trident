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
    "num_bt, num_ch, vec_sz, use_inp", [(5, 10, 1024, True), (2, 2, 25000, False)]
)
def test_forward(num_bt, num_ch, vec_sz, use_inp, dtype, device):
    ctor_args = {"device": device, "dtype": dtype}
    inp = torch.randn(num_bt, num_ch, vec_sz, **ctor_args)

    assert util.equal(
        torch.nn.functional.instance_norm(inp), trident.function.instance_norm(inp)
    )

    run_mean = torch.ones(num_ch, **ctor_args)
    run_var = torch.zeros(num_ch, **ctor_args)

    def inference(func):
        i = run_mean.clone()
        j = run_var.clone()
        return [func(inp, i, j, use_input_stats=use_inp), i, j]

    (x, y, z) = inference(torch.nn.functional.instance_norm)
    (a, b, c) = inference(trident.function.instance_norm)

    assert util.equal(x, a)
    assert util.equal(y, b)
    assert util.equal(z, c)

    wgt = torch.randn(num_ch, **ctor_args)
    bis = torch.randn(num_ch, **ctor_args)

    assert util.equal(
        torch.nn.functional.instance_norm(inp, weight=wgt, bias=bis),
        trident.function.instance_norm(inp, weight=wgt, bias=bis),
    )

    def inference(func):
        i = run_mean.clone()
        j = run_var.clone()
        return [func(inp, i, j, wgt, bis, use_input_stats=use_inp), i, j]

    (x, y, z) = inference(torch.nn.functional.instance_norm)
    (a, b, c) = inference(trident.function.instance_norm)

    assert util.equal(x, a)
    assert util.equal(y, b)
    assert util.equal(z, c)


@pytest.mark.parametrize("num_bt, num_ch, vec_sz", [(5, 10, 1024), (2, 2, 25000)])
def test_backward(num_bt, num_ch, vec_sz, device):
    ctor_args = {"device": device}
    inp = torch.randn(num_bt, num_ch, vec_sz, **ctor_args)
    tgt = torch.randn(num_bt, num_ch, vec_sz, **ctor_args)

    def train(func):
        i = inp.clone()
        i.requires_grad = True
        func(i).backward(tgt, retain_graph=True)
        return [i.grad]

    (x,) = train(torch.nn.functional.instance_norm)
    (a,) = train(trident.function.instance_norm)

    assert util.equal(x, a)

    wgt = torch.zeros(num_ch, **ctor_args).zero_()
    bis = torch.zeros(num_ch, **ctor_args).fill_(1)

    def train(func):
        i = inp.clone()
        j = wgt.clone()
        k = bis.clone()
        i.requires_grad = j.requires_grad = k.requires_grad = True
        func(i, weight=j, bias=k).backward(tgt, retain_graph=True)
        return [i.grad, j.grad, k.grad]

    (x, y, z) = train(torch.nn.functional.instance_norm)
    (a, b, c) = train(trident.function.instance_norm)

    assert util.equal(x, a)
    assert util.equal(y, b)
    assert util.equal(z, c)


@pytest.mark.parametrize("num_bt, num_ch, w, h", [(1, 1, 64, 64)])
def test_module(num_bt, num_ch, w, h, dtype, device):
    ctor_args = {"device": device, "dtype": dtype}
    inp = torch.randn(num_bt, num_ch, w, h, **ctor_args)

    inst_norm2d = trident.InstanceNorm2d(num_ch, **ctor_args)

    assert inst_norm2d.forward(inp) is not None

    inst_norm2d = trident.InstanceNorm2d(num_ch, affine=True, **ctor_args)

    assert inst_norm2d.forward(inp) is not None

    inst_norm2d = trident.InstanceNorm2d(num_ch, track_running_stats=True, **ctor_args)

    assert inst_norm2d.forward(inp) is not None

    inst_norm2d = trident.InstanceNorm2d(
        num_ch, affine=False, track_running_stats=True, **ctor_args
    )

    assert inst_norm2d.forward(inp) is not None
