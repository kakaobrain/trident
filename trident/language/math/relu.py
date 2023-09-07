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


class ReLU:
    @staticmethod
    @triton.jit
    def forward(input: tl.tensor):
        return tl.where(input > 0, input, 0)

    @staticmethod
    @triton.jit
    def backward(grad_output: tl.tensor, input: tl.tensor):
        return tl.where(input > 0, grad_output, 0)
