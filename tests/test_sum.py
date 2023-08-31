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


@pytest.mark.parametrize("y_size, x_size, dim", [(1000, 2000, 0), (2000, 1000, 1)])
def test_forward(y_size, x_size, dim, device):
    input = torch.randn(y_size, x_size, device=device)

    assert util.equal(torch.sum(input, dim), trident.function.sum(input, dim))


@pytest.mark.parametrize("y_size, x_size, dim", [(2000, 1000, 0), (1000, 2000, 1)])
def test_backward(y_size, x_size, dim, device):
    input = torch.randn(y_size, x_size, device=device)
    target = torch.randn(x_size if dim == 0 else y_size, device=device)

    def train(func):
        i = input.clone()
        i.requires_grad = True
        func(i, dim).backward(target, retain_graph=True)
        return [i.grad]

    (x,) = train(torch.sum)
    (a,) = train(trident.function.sum)

    assert util.equal(x, a)


@pytest.mark.parametrize("dim", [0, 1])
def test_sum_issue1(dim, device):
    factory_kwargs = {"device": device, "dtype": torch.float16}
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

    assert util.equal(torch.sum(input, dim), trident.function.sum(input, dim))
