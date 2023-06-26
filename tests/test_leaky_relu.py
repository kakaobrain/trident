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
    x = torch.randn(num_batches, num_elements, dtype=dtype, device='cuda')

    assert util.equal(torch.nn.functional.leaky_relu(x), trident.function.leaky_relu(x))


@pytest.mark.parametrize('num_batches, num_elements', [(512, 512), (200, 300), (100, 1), (1, 100)])
def test_backward(num_batches, num_elements):
    x = torch.randn(num_batches, num_elements, device='cuda')
    y = torch.randn(num_batches, num_elements, device='cuda')

    a = x.clone()
    b = x.clone()
    a.requires_grad = b.requires_grad = True

    util.train(a, y, torch.nn.LeakyReLU())
    util.train(b, y, trident.LeakyReLU())

    assert util.equal(a.grad, b.grad)
