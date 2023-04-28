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
import triton
from trident import kernel


def leaky_relu(x, a=0.01):
    """
    Applies a leaky relu function to the input tensor and return the result.

    :param x: Input tensor.
    :param a: Controls the angle of the negative slope.
    :return: Output tensor.
    """
    assert x.is_cuda and x.is_contiguous()

    size_0, size_1 = x.shape
    y = torch.empty_like(x)

    assert y.is_cuda and x.is_contiguous

    def get_block_size():
        return min(triton.next_power_of_2(size_1), 1 << 14)

    grid = lambda meta: (size_0, triton.cdiv(size_1, meta['block_size']), )
    kernel.leaky_relu_forward[grid](x, x.stride(0), x.stride(1),
                                    y, y.stride(0), y.stride(1),
                                    a, size_1,
                                    block_size=get_block_size())

    return y
