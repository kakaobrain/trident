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


@pytest.mark.parametrize("num_batches, y_size, x_size", [(2, 5, 1), (3, 16, 40)])
def test_forward(num_batches, y_size, x_size, device):
    input = torch.randn(num_batches, y_size, x_size, device=device)

    assert util.equal(
        torch.nn.functional.batch_norm(input, running_mean=None, running_var=None, training=True),
        trident.function.batch_norm(input, running_mean=None, running_var=None, training=True),
    )

    running_mean = torch.ones(y_size, device=device)
    running_var = torch.zeros(y_size, device=device)

    def inference(func):
        i = running_mean.clone()
        j = running_var.clone()
        return func(input, i, j, training=True), i, j

    (x, y, z) = inference(torch.nn.functional.batch_norm)
    (a, b, c) = inference(trident.function.batch_norm)

    assert util.equal(x, a)
    assert util.equal(y, b)
    assert util.equal(z, c)


@pytest.mark.parametrize("num_batches, y_size, x_size", [(3, 10, 5), (11, 40, 1)])
def test_backward(num_batches, y_size, x_size, device):
    input = torch.randn(num_batches, y_size, x_size, device=device)
    weight = torch.randn(y_size, device=device)
    bias = torch.randn(y_size, device=device)
    grad_output = torch.randn(num_batches, y_size, x_size, device=device)

    def train(func):
        i = input.clone()
        j = weight.clone()
        k = bias.clone()
        i.requires_grad = j.requires_grad = k.requires_grad = True
        func(i, weight=j, bias=k, running_mean=None, running_var=None, training=True).backward(
            grad_output, retain_graph=True
        )
        return i.grad, j.grad, k.grad

    (x, y, z) = train(torch.nn.functional.batch_norm)
    (a, b, c) = train(trident.function.batch_norm)

    assert util.equal(x, a)
    assert util.equal(y, b)
    assert util.equal(z, c)


@pytest.mark.parametrize("num_batches, y_size", [(3, 20)])
def test_batch_norm(num_batches, y_size, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(num_batches, y_size, **factory_kwargs)

    output = trident.BatchNorm1d(y_size, **factory_kwargs).forward(input)
    assert output is not None and output.dtype == dtype
