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


def test_function(input_3d):
    assert utility.equal(torch.nn.functional.instance_norm(input_3d), trident.function.instance_norm(input_3d))


def test_module_2d(input_3d, input_4d):
    assert utility.equal(torch.nn.InstanceNorm2d(64).forward(input_3d), trident.InstanceNorm2d(64).forward(input_3d))
    assert utility.equal(torch.nn.InstanceNorm2d(64).forward(input_4d), trident.InstanceNorm2d(64).forward(input_4d))