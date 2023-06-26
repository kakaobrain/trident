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


@pytest.mark.parametrize('num_batches, num_elements', [(512, 512), (200, 300), (100, 1), (1, 100)])
def test_forward(num_batches, num_elements, dtype):
    w = torch.randn(num_elements, dtype=dtype, device='cuda')
    x = torch.randn((num_batches, num_elements), dtype=dtype, device='cuda', requires_grad=True)
    assert util.equal(
        torch.nn.functional.prelu(x, w), trident.function.prelu(x, w)
    )


@pytest.mark.parametrize('num_batches, num_elements', [(512, 512), (200, 300), (100, 1), (1, 100)])
def test_backward(num_batches, num_elements):
    x = torch.randn((num_batches, num_elements), device='cuda')
    y = torch.randn((num_batches, num_elements), device='cuda')

    a = x.clone()
    b = x.clone()
    a.requires_grad = b.requires_grad = True

    prelu_a = torch.nn.PReLU(num_elements, 0.3, device='cuda')
    prelu_b = trident.PReLU(num_elements, 0.3)

    util.train(a, y, prelu_a)
    util.train(b, y, prelu_b)

    assert util.equal(a.grad, b.grad)
    assert util.equal(prelu_a.weight.grad, prelu_b.weight.grad)
