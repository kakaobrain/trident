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


def test_function(input2d):
    assert util.equal(
        torch.nn.functional.softmax(input2d, 1), trident.function.softmax(input2d, 1)
    )


def test_forward(input2d):
    assert util.equal(torch.nn.Softmax(1).forward(input2d), trident.Softmax(1).forward(input2d))


@pytest.mark.parametrize("num_batches, num_elements", [(5, 32), (4, 64), (3, 128), (2, 256), (1, 512)])
def test_backward(num_batches, num_elements):
    t = torch.randn(num_batches, num_elements, device='cuda')
    x = torch.randn(num_batches, num_elements, device='cuda')

    a = x.clone()
    b = x.clone()
    a.requires_grad = b.requires_grad = True

    util.train(a, t, torch.nn.Softmax(1), criterion=torch.nn.CrossEntropyLoss())
    util.train(b, t, trident.Softmax(1), criterion=torch.nn.CrossEntropyLoss())

    assert util.equal(a.grad, b.grad)
