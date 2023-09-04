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


@pytest.mark.parametrize("num_vec, vec_sz, training", [(1, 5, False), (3, 16, False), (7, 30, True)])
def test_function(num_vec, vec_sz, training, device):
    inp = torch.randn(num_vec, vec_sz, device=device)
    m = torch.zeros(vec_sz, device=device)
    v = torch.ones(vec_sz, device=device)
    assert util.equal(
        torch.nn.functional.batch_norm(inp, m, v, training=training),
        trident.function.batch_norm(inp, m, v, training=training),
    )


@pytest.mark.parametrize("num_vec, vec_sz, afn", [(3, 20, True), (7, 99, False)])
def test_forward(num_vec, vec_sz, afn, device):
    inp = torch.randn(num_vec, vec_sz, device=device)
    lyr0 = torch.nn.BatchNorm1d(vec_sz, affine=afn, device=device)
    lyr1 = trident.BatchNorm1d(vec_sz, affine=afn, device=device)
    assert util.equal(lyr0.forward(inp), lyr1.forward(inp))
    assert util.equal(lyr0.running_mean, lyr1.running_mean)
    assert util.equal(lyr0.running_var, lyr1.running_var)
    assert util.equal(lyr0.forward(inp), lyr1.forward(inp))
    assert util.equal(lyr0.running_mean, lyr1.running_mean)
    assert util.equal(lyr0.running_var, lyr1.running_var)


@pytest.mark.parametrize("num_vec, vec_sz, afn", [(3, 10, False), (11, 40, True)])
def test_backward(num_vec, vec_sz, afn, device):
    inp = torch.randn(num_vec, vec_sz, device=device)
    tgt = torch.randn(num_vec, vec_sz, device=device)

    a = inp.clone()
    b = inp.clone()
    a.requires_grad = b.requires_grad = True

    lyr0 = torch.nn.BatchNorm1d(vec_sz, affine=afn, device=device)
    lyr1 = trident.BatchNorm1d(vec_sz, affine=afn, device=device)

    util.train(a, tgt, lyr0)
    util.train(b, tgt, lyr1)
    assert util.equal(a.grad, b.grad)

    if afn:
        assert util.equal(lyr0.weight.grad, lyr1.weight.grad)
        assert util.equal(lyr0.bias.grad, lyr1.bias.grad)


@pytest.mark.parametrize("y_size, x_size, affine", [(3, 20, True)])
def test_batch_norm(y_size, x_size, affine, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(y_size, x_size, **factory_kwargs)

    output = trident.BatchNorm1d(x_size, affine=affine, **factory_kwargs).forward(input)
    assert output is not None and output.dtype == dtype
