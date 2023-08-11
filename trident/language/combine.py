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

from trident import language


@triton.jit
def combine_welford(m2_a, mean_a, count_a, m2_b, mean_b, count_b):
    return (
        m2_a + m2_b + language.pow2(mean_b - mean_a) * count_a * count_b / (count_a + count_b),
        (mean_a * count_a + mean_b * count_b) / (count_a + count_b),
        count_a + count_b,
    )
