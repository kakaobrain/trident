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


@pytest.fixture(scope='session')
def input_2d():
    return torch.randn(1024, 8192, device='cuda', requires_grad=True)


@pytest.fixture(scope='session')
def input_3d():
    return torch.randn(4, 512, 512, device='cuda', requires_grad=True)


@pytest.fixture(scope='session')
def input_4d():
    return torch.randn(32, 4, 256, 256, device='cuda', requires_grad=True)


@pytest.fixture(scope='session')
def target():
    return torch.randn(1024, 8192, device='cuda')


@pytest.fixture(scope='session')
def weight():
    return torch.randn(1024, 8192, device='cuda', requires_grad=True)


@pytest.fixture(scope='session')
def bias():
    return torch.randn(1024, device='cuda', requires_grad=True)
