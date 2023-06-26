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


def test_function(dtype):
    inp = torch.randn(512, 512, dtype=dtype, device='cuda', requires_grad=True)
    assert util.equal(torch.nn.functional.relu(inp), trident.function.relu(inp))


def test_forward(dtype):
    inp = torch.randn(512, 512, dtype=dtype, device='cuda', requires_grad=True)
    assert util.equal(torch.nn.ReLU().forward(inp), trident.ReLU().forward(inp))


@pytest.mark.parametrize("num_batches, num_elements", [(5, 32), (4, 64), (3, 128), (2, 256), (1, 512)])
def test_backward(num_batches, num_elements):
    x = torch.randn(num_batches, num_elements, device='cuda')
    y = torch.randn(num_batches, num_elements, device='cuda')

    a = x.clone()
    b = x.clone()
    a.requires_grad = b.requires_grad = True

    util.train(a, y, torch.nn.ReLU())
    util.train(b, y, trident.ReLU())

    assert util.equal(a.grad, b.grad)
