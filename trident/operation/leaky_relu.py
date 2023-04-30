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
from trident import kernel, function


class LeakyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a):
        """
        Applies a leaky relu function to the input tensor and return the result.

        :param ctx: Context object. It's used to store information for the backpropagation.
        :param x: Input tensor.
        :param a: Controls the angle of the negative slope.
        :return: Output tensor.
        """
        y = function.leaky_relu(x, a)
        ctx.save_for_backward(x)
        ctx.a = a

        return y

    @staticmethod
    def backward(ctx, *grad_outputs):
        x = ctx.saved_tensors[0]
        a = ctx.a

        assert x.is_cuda and x.is_contiguous()

        x_shape_0, x_shape_1 = x.shape
        d = torch.empty_like(x)

        assert d.is_cuda and d.is_contiguous

        def get_block_size():
            return min(triton.next_power_of_2(x_shape_1), 1 << 14)

        grid = lambda meta: (x_shape_0, triton.cdiv(x_shape_1, meta['block_size']), )
        kernel.leaky_relu_backward[grid](x, x.stride(0), x.stride(1),
                                         d, d.stride(0), d.stride(1),
                                         a, x_shape_1,
                                         block_size=get_block_size())

        return d, None
