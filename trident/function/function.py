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

from trident import operation


def adaptive_avg_pool2d(input, output_size):
    """
    Applies Adaptive Average Pooling 2D to an input.

    See AdaptiveAvgPool2d for details.
    """
    return operation.AdaptiveAvgPool2d.apply(input, output_size)


def instance_norm(input, eps=1e-05):
    """
    Applies Instance Normalization for each channel in each data sample in a batch.

    See InstanceNorm2d for details.
    """
    return operation.InstanceNorm.apply(input, eps)


def leaky_relu(input, negative_slope=0.01):
    """
    Applies Leaky ReLU to an input.

    See LeakyReLU for more details.
    """
    return operation.LeakyReLU.apply(input, negative_slope)


def linear(input, weight, bias=None, activation=''):
    """
    Applies Linear Transformation to an input.

    See Linear for more details.
    """
    return operation.Linear.apply(input, weight, bias, activation)


def max_pool2d(input, kernel_size):
    """
    Applies Max Pooling 2D to an input.

    See MaxPool2d for details.
    """
    return operation.MaxPool2d.apply(input, kernel_size)


def relu(input):
    """
    Applies ReLU to an input.

    See ReLU for more details.
    """
    return operation.ReLU.apply(input)


def prelu(input, negative_slopes):
    """
    Applies PReLU to an input.

    See PReLU for more details.
    """
    return operation.PReLU.apply(input, negative_slopes)


def softmax(input, dim=None):
    """
    Applies Softmax to an input rescaling them so that an output lie in the range [0,1] and sum to 1.

    See Softmax for more details.
    """
    return operation.Softmax.apply(input, dim)
