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

import torch

from trident import operation


class AdaptiveAvgPool2d(torch.nn.Module):
    def __init__(self, output_size):
        """
        Applies Adaptive Average Pooling 2D to an input.

        Args:
            output_size: the target output size.
        """
        super().__init__()

        self.output_size = output_size

    def forward(self, input):
        """
        Applies Adaptive Average Pooling 2D to an input.

        Args:
            input: an input (N, C, H, W) or (C, H, W)

        Returns:
            an output (N, C, T, T) or (C, T, T)
        """
        assert input.dim() == 3 or input.dim() == 4

        x = AdaptiveAvgPool2d.__view(input)
        y = operation.AdaptiveAvgPool2d.apply(x, self.output_size)

        return y if y.dim() == 4 else y.squeeze()

    @staticmethod
    def __shape(x):
        if x.dim() == 3:
            return 1, x.shape[0], x.shape[1], x.shape[2]
        else:
            return x.shape

    @staticmethod
    def __view(x):
        num_batches, num_channels, height, width = AdaptiveAvgPool2d.__shape(x)
        return x.view(num_batches, num_channels, height, width)


class Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        """
        Applies Convolution 2D to an input.

        Args:
            in_channels: number of channels in the input image
            out_channels: number of channels produced by the convolution
            kernel_size: size of the convolution kernel
            bias: If True, adds a learnable bias to the output.
        """
        super().__init__()

        self.in_channels = in_channels
        self.weight = torch.empty(out_channels, in_channels, kernel_size, kernel_size, device='cuda', dtype=torch.float)
        self.bias = torch.empty(out_channels, device='cuda', dtype=torch.float) if bias else None

    def forward(self, input):
        """
        Applies Convolution 2D to an input.

        Args:
            input: an input (N, C, H, W) or (C, H, W)

        Returns:
            an output (N, C, R, C) or (C, R, C)
        """
        return operation.Conv2d.apply(input, self.weight, self.bias)


class InstanceNorm2d(torch.nn.Module):
    def __init__(self, num_features, eps=1e-05):
        """
        Applies Instance Normalization to an input as described in the paper Instance Normalization: The Missing
        Ingredient for Fast Stylization.

        Args:
            num_features: C from an expected input of size (N,C,H,W) or (C,H,W)
            eps: a value added to the denominator for numerical stability
        """
        super().__init__()

        self.eps = eps

    def forward(self, input):
        """
        Applies Instance Normalization to an input.

        Args:
            input: an input (N,C,H,W) or (C,H,W)

        Returns:
            an output with the same dimension and shape as an input
        """
        assert input.dim() == 3 or input.dim() == 4

        x = InstanceNorm2d.__view(input)
        y = operation.InstanceNorm.apply(x, self.eps)

        return y.view(input.shape)

    @staticmethod
    def __shape(x):
        if x.dim() == 3:
            return 1, x.shape[0], x.shape[1], x.shape[2]
        else:
            return x.shape

    @staticmethod
    def __view(x):
        num_batches, num_channels, height, width = InstanceNorm2d.__shape(x)
        return x.view(num_batches, num_channels, height * width)


class LeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=1e-2):
        """
        Applies Leaky ReLU to an input.

        Args:
            negative_slope: Controls the angle of the negative slope(which is used for negative input values)
        """
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        """
        Applies Leaky ReLU to an input.

        Args:
            input: an input

        Returns:
            an output with the same dimension and shape as an input
        """
        return operation.LeakyReLU.apply(input, self.negative_slope)


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        """
        Applies Linear Transformation to an input.

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            bias: If set to False, the layer will not learn an additive bias
        """
        super().__init__()

        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features, device='cuda'))
        self.bias = torch.nn.Parameter(torch.randn(out_features, device='cuda')) if bias else None

    def forward(self, input):
        """
        Applies Linear Transformation to an input.

        Args:
            input: an input (*, in_features)

        Returns:
            an output (*, out_features)
        """
        return operation.Linear.apply(input, self.weight, self.bias)


class MaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size):
        """
        Applies Max Pooling 2D to an input.

        Args:
            kernel_size: the size of the window to take a max over
        """
        super().__init__()

        self.kernel_size = kernel_size

    def forward(self, input):
        """
        Applies Max Pooling 2D to an input.

        Args:
            input: an input (N, C, inp_height, inp_width)

        Returns:
            an output (N, C, out_height, out_width)
        """
        return operation.MaxPool2d.apply(input, self.kernel_size)


class ReLU(torch.nn.Module):
    def __init__(self):
        """
        Applies Leaky ReLU to an input.
        """
        super().__init__()

    def forward(self, input):
        """
        Applies Leaky ReLU to an input.

        Args:
            input: an input

        Returns:
            an output with the same dimension and shape as an input
        """
        return operation.ReLU.apply(input)


class PReLU(torch.nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        """
        Applies PReLU to an input.

        Args:
            num_parameters: number of weight to learn
            init: the initial value of weight
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(num_parameters, device='cuda').fill_(init))

    def forward(self, input):
        """
        Applies PReLU to an input.

        Args:
            input: an input

        Returns:
            an output with the same dimension and shape as an input
        """
        return operation.PReLU.apply(input, self.weight)


class Softmax(torch.nn.Module):
    def __init__(self, dim=None):
        """
        Applies Softmax to an input rescaling them so that an output lie in the range [0,1] and sum to 1.

        Args:
            dim: A dimension along which Softmax will be computed (so every slice along dim will sum to 1)
        """
        super().__init__()

        self.dim = dim

    def forward(self, input):
        """
        Applies Softmax to input.

        Args:
            input: an input

        Returns:
            an output with the same dimension and shape as an input with values in the range [0, 1]
        """
        return operation.Softmax.apply(input, self.dim)
