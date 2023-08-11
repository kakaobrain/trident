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
    "num_batches, num_channels, length, use_input_stats",
    [(5, 10, 1024, True), (2, 2, 25000, False)],
)
def test_forward(num_batches, num_channels, length, use_input_stats, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(num_batches, num_channels, length, **factory_kwargs)

    assert util.equal(torch.nn.functional.instance_norm(input), trident.function.instance_norm(input))

    run_mean = torch.ones(num_channels, **factory_kwargs)
    run_var = torch.zeros(num_channels, **factory_kwargs)

    def inference(func):
        i = run_mean.clone()
        j = run_var.clone()
        return [func(input, i, j, use_input_stats=use_input_stats), i, j]

    (x, y, z) = inference(torch.nn.functional.instance_norm)
    (a, b, c) = inference(trident.function.instance_norm)

    assert util.equal(x, a)
    assert util.equal(y, b)
    assert util.equal(z, c)

    weight = torch.randn(num_channels, **factory_kwargs)
    bias = torch.randn(num_channels, **factory_kwargs)

    assert util.equal(
        torch.nn.functional.instance_norm(input, weight=weight, bias=bias),
        trident.function.instance_norm(input, weight=weight, bias=bias),
    )

    def inference(func):
        i = run_mean.clone()
        j = run_var.clone()
        return [func(input, i, j, weight, bias, use_input_stats=use_input_stats), i, j]

    (x, y, z) = inference(torch.nn.functional.instance_norm)
    (a, b, c) = inference(trident.function.instance_norm)

    assert util.equal(x, a)
    assert util.equal(y, b)
    assert util.equal(z, c)


@pytest.mark.parametrize("num_batches, num_channels, length", [(5, 10, 1024), (2, 2, 25000)])
def test_backward(num_batches, num_channels, length, device):
    factory_kwargs = {"device": device}
    input = torch.randn(num_batches, num_channels, length, **factory_kwargs)
    target = torch.rand_like(input)

    def train(func):
        i = input.clone()
        i.requires_grad = True
        func(i).backward(target, retain_graph=True)
        return [i.grad]

    (x,) = train(torch.nn.functional.instance_norm)
    (a,) = train(trident.function.instance_norm)

    assert util.equal(x, a)

    weight = torch.zeros(num_channels, **factory_kwargs).zero_()
    bias = torch.zeros(num_channels, **factory_kwargs).fill_(1)

    def train(func):
        i = input.clone()
        j = weight.clone()
        k = bias.clone()
        i.requires_grad = j.requires_grad = k.requires_grad = True
        func(i, weight=j, bias=k).backward(target, retain_graph=True)
        return [i.grad, j.grad, k.grad]

    (x, y, z) = train(torch.nn.functional.instance_norm)
    (a, b, c) = train(trident.function.instance_norm)

    assert util.equal(x, a)
    assert util.equal(y, b)
    assert util.equal(z, c)


@pytest.mark.parametrize("num_channels, length", [(1, 64)])
def test_instance_norm1d(num_channels, length, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(num_channels, length, **factory_kwargs)

    operation = trident.InstanceNorm1d(num_channels, **factory_kwargs)
    assert operation.forward(input) is not None

    operation = trident.InstanceNorm1d(num_channels, affine=True, **factory_kwargs)
    assert operation.forward(input) is not None

    operation = trident.InstanceNorm1d(num_channels, track_running_stats=True, **factory_kwargs)
    assert operation.forward(input) is not None

    operation = trident.InstanceNorm1d(num_channels, affine=False, track_running_stats=True, **factory_kwargs)
    assert operation.forward(input) is not None


@pytest.mark.parametrize("num_batches, num_channels, height, width", [(1, 1, 64, 64)])
def test_instance_norm2d(num_batches, num_channels, height, width, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(num_batches, num_channels, height, width, **factory_kwargs)

    operation = trident.InstanceNorm2d(num_channels, **factory_kwargs)
    assert operation.forward(input) is not None

    operation = trident.InstanceNorm2d(num_channels, affine=True, **factory_kwargs)
    assert operation.forward(input) is not None

    operation = trident.InstanceNorm2d(num_channels, track_running_stats=True, **factory_kwargs)
    assert operation.forward(input) is not None

    operation = trident.InstanceNorm2d(num_channels, affine=False, track_running_stats=True, **factory_kwargs)
    assert operation.forward(input) is not None
