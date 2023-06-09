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


@pytest.mark.parametrize('target_size', [2, 4, 8])
def test_function(input4d, target_size):
    assert util.equal(
        torch.nn.functional.adaptive_avg_pool2d(input4d, target_size),
        trident.function.adaptive_avg_pool2d(input4d, target_size)
    )


@pytest.mark.parametrize('target_size', [2, 4, 8])
def test_module(input3d, target_size):
    assert util.equal(
        torch.nn.AdaptiveAvgPool2d(target_size).forward(input3d),
        trident.AdaptiveAvgPool2d(target_size).forward(input3d)
    )
