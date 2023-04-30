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
from trident import operation


class LeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=1e-2):
        """
        Applies a leaky relu function to the input tensor and return the result.

        :param negative_slope: Controls the angle of the negative slope.
        """
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, x):
        return operation.LeakyReLU.apply(x, self.negative_slope)


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


class Softmax(torch.nn.Module):
    def __init__(self, dim=None):
        """
        Applies a softmax function to the input tensor. Output tensor in the range [0,1] and sum to 1.

        :param dim: A dimension along which softmax will be computed.
        """
        super().__init__()

        self.dim = dim

    def forward(self, x):
        return operation.Softmax.apply(x, self.dim)
