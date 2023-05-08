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

from trident import operation


def instance_norm(x, eps=1e-05):
    """
    Applies Instance Normalization for each channel in each data sample in a batch.

    :param x: Input tensor.
    :param eps: Epsilon.
    :return: Output tensor.
    """
    return operation.InstanceNorm.apply(x, eps)


def leaky_relu(x, a=0.01):
    """
    Applies a leaky relu function to the input tensor and return the result.

    :param x: Input tensor.
    :param a: Controls the angle of the negative slope.
    :return: Output tensor.
    """
    return operation.LeakyReLU.apply(x, a)


def linear(x, w, b=None, activation=''):
    """
    Applies a linear transformation on the input tensor x using the weight tensor w
    and the bias tensor b, and returns the result.

    :param x: Input tensor. The tensor shape is (*, in_features).
    :param w: Weight tensor. The tensor shape is (out_features, in_features).
    :param b: Bias tensor. The tensor shape is (out_features).
    :param activation: Activation function. Supports for relu and leaky_relu.
    :return: Output tensor. The tensor shape is (*,out_features).
    """
    return operation.Linear.apply(x, w, b, activation)


def softmax(x, dim=None):
    """
    Applies a softmax function to the input tensor and return the result.

    :param x: Input tensor.
    :param dim: A dimension along which softmax will be computed.
    :return: Output tensor.
    """
    return operation.Softmax.apply(x, dim)
