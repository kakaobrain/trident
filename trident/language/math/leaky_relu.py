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


class LeakyReLU:
    @staticmethod
    @triton.jit
    def forward(input: tl.tensor, negative_slope: tl.float32):
        condition = input > 0
        return tl.where(condition, input, 0) + negative_slope * tl.where(condition, 0, input)

    @staticmethod
    @triton.jit
    def backward(grad_output: tl.tensor, input: tl.tensor, negative_slope: tl.float32):
        return grad_output * tl.where(input > 0, 1, negative_slope)
