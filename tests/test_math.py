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
import triton

import trident
from tests import util


@pytest.mark.parametrize("height, width, axis", [(20000, 20000, 0), (20000, 20000, 1)])
def test_sum(height, width, axis, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}

    if axis == 0:
        output_size = width
        size_along_axis = height
    else:
        output_size = height
        size_along_axis = width

    input = torch.randn(height, width, **factory_kwargs)
    output = torch.empty(output_size, **factory_kwargs)

    def grid(meta):
        return [output_size]

    trident.kernel.sum[grid](
        output,
        input,
        height,
        width,
        axis,
        trident.util.block_size(size_along_axis, input.element_size()),
        trident.util.dtype(dtype),
    )

    assert util.equal(torch.sum(input, dim=axis), output)


@pytest.mark.parametrize("axis", [0, 1])
def test_sum_issue1(axis, device):
    dtype = torch.float16
    factory_kwargs = {"device": device, "dtype": dtype}

    input = torch.tensor(
        [
            [
                30.6875,
                -40.4375,
                -29.1719,
                81.1875,
                23.3125,
                3.6348,
                6.0508,
                -100.5000,
                -6.0273,
                11.6562,
            ],
            [
                21.5469,
                11.3438,
                14.0000,
                33.7188,
                13.4844,
                -18.0938,
                27.5156,
                -29.0625,
                -1.7559,
                20.8594,
            ],
            [
                28.6406,
                -30.1094,
                22.6406,
                -35.8750,
                3.5410,
                -66.1250,
                15.6016,
                -22.4375,
                50.0625,
                39.6562,
            ],
            [
                5.3281,
                -75.1875,
                -13.3828,
                -39.9688,
                -59.9062,
                14.7812,
                -23.0625,
                -3.4336,
                -34.8125,
                32.7812,
            ],
            [
                20.1406,
                -33.4375,
                -50.3438,
                -25.2812,
                69.6250,
                2.2090,
                18.9062,
                16.3750,
                -7.9922,
                27.1562,
            ],
        ],
        **factory_kwargs,
    )
    height, width = input.shape

    if axis == 0:
        output_size = width
        size_along_axis = height
    else:
        output_size = height
        size_along_axis = width

    output = torch.empty(output_size, **factory_kwargs)

    def grid(meta):
        return [output_size]

    trident.kernel.sum[grid](
        output,
        input,
        height,
        width,
        axis,
        trident.util.block_size(size_along_axis, input.element_size()),
        trident.util.dtype(dtype),
    )

    assert util.equal(torch.sum(input, dim=axis), output)
