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


@pytest.mark.parametrize("kernel_size", [2, 3, 4, 5, 6, 7, 8])
def test_function(kernel_size, device):
    input = torch.randn(2, 3, 128, 128, device=device)

    assert util.equal(
        torch.nn.functional.max_pool2d(input, kernel_size),
        trident.function.max_pool2d(input, kernel_size),
    )


@pytest.mark.parametrize("kernel_size", [32, 64, 96, 128])
def test_forward(kernel_size, device):
    input = torch.randn(10, 7, 256, 256, device=device)

    assert util.equal(
        torch.nn.MaxPool2d(kernel_size).forward(input),
        trident.MaxPool2d(kernel_size).forward(input),
    )


@pytest.mark.parametrize("kernel_size", [32])
def test_max_pool2d(kernel_size, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    input = torch.randn(2, 3, 128, 128, **factory_kwargs)

    output = trident.MaxPool2d(kernel_size).forward(input)
    assert output is not None and output.dtype == dtype
