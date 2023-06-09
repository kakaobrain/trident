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


def test_function(input3d):
    assert util.equal(torch.nn.functional.instance_norm(input3d), trident.function.instance_norm(input3d))


def test_module_2d(input3d, input4d):
    assert util.equal(torch.nn.InstanceNorm2d(2).forward(input3d), trident.InstanceNorm2d(2).forward(input3d))
    assert util.equal(torch.nn.InstanceNorm2d(2).forward(input4d), trident.InstanceNorm2d(2).forward(input4d))
