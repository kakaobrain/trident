"""
Copyright ⓒ Kakao Brain Corp. All rights reserved.

Unauthorized copying of this file, via any medium is strictly prohibited.
Proprietary and confidential.
"""

import torch
import operation


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        """
        Applies a linear transformation to the incoming data.
        Input shape is (*, in_features) and output shape is (*, out_features).

        :param in_features: Size of each input sample.
        :param out_features: Size of each output sample
        :param bias: If set to False, the layer will not learn an additive bias. Default is True.
        """
        super().__init__()

        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features, device='cuda'))
        self.bias = torch.nn.Parameter(torch.randn(out_features, device='cuda')) if bias else None

    def forward(self, x):
        return operation.Linear.apply(x, self.weight, self.bias)
