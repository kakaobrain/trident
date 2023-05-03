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

import triton

from trident import language

@triton.jit
def instance_norm_forward(x_ptr, x_batch_stride, x_channel_stride,
                          y_ptr, y_batch_stride, y_channel_stride,
                          num_elements, eps,
                          block_size: triton.language.constexpr):
    batch = triton.language.program_id(0)
    channel = triton.language.program_id(1)

    block = triton.language.arange(0, block_size)
    mask = block < num_elements

    x_ptr += batch * x_batch_stride + channel * x_channel_stride
    y_ptr += batch * y_batch_stride + channel * y_channel_stride

    x = triton.language.load(x_ptr + block, mask, 0.0)

    mean = triton.language.sum(x, 0) / num_elements
    var = language.var(mask, x, num_elements, mean, correction=0)
    y = (x - mean) / triton.language.sqrt(var + eps)

    triton.language.store(y_ptr + block, y, mask)
