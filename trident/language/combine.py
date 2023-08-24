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

import triton
import triton.language as tl

from trident import language


@triton.jit
def combine_welford(m2_a, mean_a, count_a, m2_b, mean_b, count_b):
    count = count_a + count_b
    return (
        m2_a + m2_b + tl.math.pow(mean_b - mean_a, 2.0) * count_a * count_b / count,
        (mean_a * count_a + mean_b * count_b) / count,
        count,
    )


@triton.jit
def combine_softmax(max_a: tl.tensor, sum_a: tl.tensor, max_b: tl.tensor, sum_b: tl.tensor):
    max = tl.math.max(max_a, max_b)
    sum = sum_a * tl.math.exp(max_a - max) + sum_b * tl.math.exp(max_b - max)
    return max, sum
