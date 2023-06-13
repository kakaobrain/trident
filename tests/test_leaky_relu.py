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


@pytest.mark.parametrize("shapes",
                         [[(512, 512), (512, 1024), (1024, 512), (100, 200), (1, 1), (1, 100), (100, 1), (1025, 1)]])
def test_forward(shapes):
    for shape in shapes:
        input = torch.randn(shape, device='cuda', requires_grad=True)
        assert util.equal(
            torch.nn.functional.leaky_relu(input), trident.function.leaky_relu(input)
        ), f'test fail on input shape:{input.shape}'


@pytest.mark.parametrize("shapes",
                         [[(512, 512), (512, 1024), (1024, 512), (100, 200), (1, 1), (1, 100), (100, 1), (1025, 1)]])
def test_backward(shapes):
    for shape in shapes:
        input = torch.randn(shape, device='cuda', requires_grad=False)
        target = torch.randn(shape, device='cuda', requires_grad=False)

        input_torch = input.clone()
        input_torch.requires_grad = True

        input_trident = input.clone()
        input_trident.requires_grad = True

        util.train(input_torch, target, torch.nn.LeakyReLU())
        util.train(input_trident, target, trident.LeakyReLU())

        assert util.equal(input_torch.grad, input_trident.grad), f'test fail on input shape:{input_torch.shape}'
