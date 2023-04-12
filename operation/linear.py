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
import function


class Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b=None):
        """
        Applies a linear transformation on the input tensor x using the weight tensor w
        and the bias tensor b, and returns the result.

        :param ctx:
        :param x: Input tensor. The tensor shape is (m, k).
        :param w: Weight tensor. The tensor shape is (n, k).
        :param b: Bias tensor. The tensor shape is (n).
        :return: Output tensor. The tensor shape is (m, n).
        """
        return function.linear(x, w, b)

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("The backward of Linear isn't implemented.")
