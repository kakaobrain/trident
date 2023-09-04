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

import math

import torch

from trident import operation


def adaptive_avg_pool2d(input, output_size):
    """
    Applies Adaptive Average Pooling 2D to an input.

    See AdaptiveAvgPool2d for details.
    """
    return operation.AdaptiveAvgPool2d.apply(input, output_size)


def argmax(input: torch.Tensor, dim: torch.int32):
    """
    Returns the indices of the maximum value of all elements in an input.
    """
    return operation.Argmax.apply(input, dim)


def batch_norm(input, running_mean=None, running_var=None, eps=1e-05, training=False):
    """
    Applies Batch Normalization for last certain number of dimensions.

    See BatchNorm for details.
    """
    if training:
        running_mean, running_var = None, None
    else:
        assert running_mean is not None and running_var is not None

    return operation.BatchNorm.apply(input, None, None, eps, running_mean, running_var)


def conv2d(input, weight, bias=None):
    """
    Applies Convolution 2D to an input.

    See Conv2d for details.
    """
    return operation.Conv2d.apply(input, weight, bias)


def cosine_similarity(x1, x2, dim=1, eps=1e-08):
    """
    Applies cosine similarity to inputs.

    See cosine similarity for detail.
    """
    return operation.CosineSimilarity.apply(x1, x2, dim, eps)


def dropout(input, p=0.5, training=True):
    """
    Applies Dropout to an input.

    See Dropout for details.
    """
    if training:
        return operation.Dropout.apply(input.view(-1, input.shape[-1]), p).view(input.shape)
    else:
        return input.clone()


def geglu(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, use_accelerator: bool = False):
    """
    Applies GEGLU to an input.

    See GEGLU for details.
    """
    output, _ = operation.GEGLU.apply(input, weight, bias, use_accelerator)
    return output


def gelu(input: torch.Tensor):
    """
    Applies the Gaussian Error Linear Units to an input.

    See GELU for details.
    """
    return operation.GELU.apply(input)


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-05):
    """
    Applies Group Normalization for last certain number of dimensions.

    See GroupNorm for details.
    """
    output, _, _ = operation.GroupNorm.apply(
        input.view(input.shape[0], input.shape[1], -1), num_groups, weight, bias, eps
    )
    return output.view(input.shape)


def instance_norm(
    input,
    running_mean=None,
    running_var=None,
    weight=None,
    bias=None,
    use_input_stats=True,
    momentum=0.1,
    eps=1e-05,
):
    """
    Applies Instance Normalization for each channel in each data sample in a batch.

    See InstanceNorm2d for details.
    """
    return operation.InstanceNorm.apply(
        input,
        running_mean,
        running_var,
        weight,
        bias,
        use_input_stats,
        momentum,
        eps,
    )


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    """
    Applies Layer Normalization for last certain number of dimensions.

    See LayerNorm for details.
    """
    output, _, _ = operation.LayerNorm.apply(input, normalized_shape, weight, bias, eps)
    return output


def leaky_relu(input, negative_slope=0.01):
    """
    Applies Leaky ReLU to an input.

    See LeakyReLU for more details.
    """
    return operation.LeakyReLU.apply(input, negative_slope)


def linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    use_accelerator: bool = False,
):
    """
    Applies Linear Transformation to an input.

    See Linear for more details.
    """
    if input.dim() == 2:
        output = operation.Linear.apply(input.view(1, *input.shape), weight, bias, use_accelerator)
        return output.view(output.shape[1:3])
    else:
        return operation.Linear.apply(input, weight, bias, use_accelerator)


def max_pool2d(input, kernel_size):
    """
    Applies Max Pooling 2D to an input.

    See MaxPool2d for details.
    """
    return operation.MaxPool2d.apply(input, kernel_size)


def max(input, dim):
    """
    Returns the max along the specified dimension in an input.

    See Max for more details.
    """
    return operation.Max.apply(input, dim)


def mean(input, dim):
    """
    Returns the mean along the specified dimension in an input.

    See Mean for more details.
    """
    return operation.Mean.apply(input, dim)


def relu(input):
    """
    Applies ReLU to an input.

    See ReLU for more details.
    """
    return operation.ReLU.apply(input)


def rms_norm(input: torch.Tensor, p: float, weight: torch.Tensor, bias: torch.Tensor = None, eps: float = 1e-08):
    """
    Applies Root Mean Square Layer Normalization to an input.

    See RMSNorm for details.
    """
    output, _ = operation.RMSNorm.apply(input.view(-1, input.shape[-1]), p, weight, bias, eps)
    return output.view(input.shape)


def shift_gelu(input: torch.Tensor, bias: torch.Tensor):
    """
    Applies shift and the Gaussian Error Linear Units to an input.

    See ShiftGELU for details.
    """
    output, _ = operation.ShiftGELU.apply(input.view(-1, input.shape[-1]), bias)
    return output.view(input.shape)


def silu(input):
    """
    Applies the Sigmoid Linear Unit to an input.

    See SiLU for more details.
    """
    return operation.SiLU.apply(input.view(-1, input.shape[-1])).view(input.shape)


def prelu(input, weight):
    """
    Applies PReLU to an input.

    See PReLU for more details.
    """
    return operation.PReLU.apply(input, weight)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    use_accelerator: bool = False,
):
    """
    Computes scaled dot product attention on query, key and value tensors,
    and applying dropout if a probability greater than 0.0 is specified.
    """
    assert len(query.size()) == 4

    return operation.Attention.apply(query, key, value, is_causal, 1.0 / math.sqrt(key.shape[-1]), use_accelerator)


def softmax(input: torch.Tensor, dim: int = None):
    """
    Applies Softmax to an input rescaling them so that an output lie in the range [0,1] and sum to 1.

    See Softmax for more details.
    """
    return operation.Softmax.apply(input, dim)


def sum(input, dim):
    """
    Returns the sum along the specified dimension in an input.

    See Sum for more details.
    """
    return operation.Sum.apply(input, dim)


def var(input, dim, correction=1):
    """
    Returns the variance along the specified dimension in an input.

    See Var for more details.
    """
    return operation.Var.apply(input, dim, correction)


def var_mean(input, dim, correction=1):
    """
    Returns the variance and mean along the specified dimension in an input.

    See VarMean for more details.
    """
    return operation.VarMean.apply(input, dim, correction)
