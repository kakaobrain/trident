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


class GeLU:
    @staticmethod
    @triton.jit
    def forward(input: tl.tensor):
        return 0.5 * input * (1 + tl.math.tanh(0.797884560802865 * (input + 0.044715 * tl.math.pow(input, 3.0))))

    @staticmethod
    @triton.jit
    def backward(input: tl.tensor):
        a = tl.math.tanh(0.797884560802865 * (input + 0.044715 * tl.math.pow(input, 3.0)))
        b = input * (1.0 - tl.math.pow(a, 2.0)) * (0.797884560802865 + 0.1070322244089 * tl.math.pow(input, 2.0))
        return 0.5 * (1.0 + a + b)
