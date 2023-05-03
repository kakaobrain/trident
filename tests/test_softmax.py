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
import triton

import trident
from tests import utility


def test_function(input_2d):
    assert triton.testing.allclose(
        torch.nn.functional.softmax(input_2d, 1), trident.function.softmax(input_2d, 1)
    )


def test_module(input_2d, target):
    x = utility.train(input_2d, target, torch.nn.Softmax(1))
    y = utility.train(input_2d, target, trident.Softmax(1))

    assert triton.testing.allclose(x, y)
    assert triton.testing.allclose(x.grad, y.grad)
