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
from tests import utility


def test_function(input_2d):
    assert utility.equal(
        torch.nn.functional.leaky_relu(input_2d), trident.function.leaky_relu(input_2d)
    )


def test_module(input_2d, target):
    x = utility.train(input_2d, target, torch.nn.LeakyReLU())
    y = utility.train(input_2d, target, trident.LeakyReLU())

    assert utility.equal(x, y)
    assert utility.equal(x.grad, y.grad)
