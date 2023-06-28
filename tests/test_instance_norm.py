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

import torch

import trident
from tests import util


def test_function(dtype):
    inp = torch.randn(2, 256, 256, dtype=dtype, device='cuda')

    assert util.equal(torch.nn.functional.instance_norm(inp), trident.function.instance_norm(inp))


def test_forward(dtype):
    inp = torch.randn(2, 256, 256, dtype=dtype, device='cuda')

    assert util.equal(torch.nn.InstanceNorm2d(2).forward(inp), trident.InstanceNorm2d(2).forward(inp))

    inp = torch.randn(4, 4, 128, 128, dtype=dtype, device='cuda')

    assert util.equal(torch.nn.InstanceNorm2d(2).forward(inp), trident.InstanceNorm2d(2).forward(inp))
