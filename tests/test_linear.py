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

import pytest
import torch

import trident
from tests import utility


@pytest.mark.skip
def test_function(input_2d, weight):
    assert utility.equal(
        torch.nn.functional.linear(input_2d, weight), trident.function.linear(input_2d, weight)
    )


@pytest.mark.skip
def test_function_with_bias(input_2d, weight, bias):
    assert utility.equal(
        torch.nn.functional.linear(input_2d, weight, bias), trident.function.linear(input_2d, weight, bias)
    )


@pytest.mark.skip
@pytest.mark.parametrize('activation', ['relu', 'leaky_relu'])
def test_function_with_activation(input_2d, weight, bias, activation):
    assert utility.equal(
        utility.activate(torch.nn.functional.linear(input_2d, weight, bias), activation),
        trident.function.linear(input_2d, weight, bias, activation)
    )
