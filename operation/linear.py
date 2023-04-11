"""
Copyright ⓒ Kakao Brain Corp. All rights reserved.

Unauthorized copying of this file, via any medium is strictly prohibited.
Proprietary and confidential.
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
