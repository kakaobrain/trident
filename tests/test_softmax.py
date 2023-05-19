"""
Copyright 2023 ⓒ Kakao Brain Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch

import trident
from tests import util


def test_function(input2d):
    assert util.equal(
        torch.nn.functional.softmax(input2d, 1), trident.function.softmax(input2d, 1)
    )


def test_module(input2d, target):
    x = util.train(input2d, target, torch.nn.Softmax(1))
    y = util.train(input2d, target, trident.Softmax(1))

    assert util.equal(x, y)
    assert util.equal(x.grad, y.grad)
