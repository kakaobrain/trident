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
from typing import Tuple

import torch

from trident import function, operation, util


class BatchNorm1d(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ):
        """
        Applies Batch Normalization over a 2D or 3D inputs.

        Args:
            num_features: number of features
            eps: a value added to the denominator for numerical stability
            momentum: the value used for the running_mean and running_var computation.
            affine: a boolean value that when set to True, this module has learnable affine parameters.
            track_running_stats: a boolean value that when set to True, this module tracks
                the running mean and variance, and when set to False, this module does not track such statistics,
                and initializes the statistics as None.
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = torch.nn.Parameter(torch.empty(num_features, **factory_kwargs).fill_(1))
            self.bias = torch.nn.Parameter(torch.zeros(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.running_mean = torch.zeros(num_features, device=device, dtype=dtype)
            self.running_var = torch.ones(num_features, device=device, dtype=dtype)
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)

    def forward(self, input: torch.Tensor):
        """
        Applies Batch Normalization to an input.

        Args:
            input: an input

        Returns:
            an output with the same dimension and shape as an input
        """
        assert input.dim() == 2
        assert input.shape[0] > 1

        if self.track_running_stats:
            self.running_mean = input.mean(axis=0) * self.momentum + self.running_mean * (1 - self.momentum)
            self.running_var = input.var(axis=0) * self.momentum + self.running_var * (1 - self.momentum)

        return operation.BatchNorm.apply(input, None, None, self.weight, self.bias, self.eps)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return (
            f"{self.num_features}, "
            f"eps={self.eps}, "
            f"momentum={self.momentum}, "
            f"affine={self.affine}, "
            f"track_running_stats={self.track_running_stats}, "
            f"backend=Trident"
        )


class CosineSimilarity(torch.nn.Module):
    def __init__(self, dim: int = 1, eps: float = 1e-8):
        """
        Applies cosine similarity to inputs.

        Args:
            dim: Dimension where cosine similarity is computed.
            eps: Small value to avoid division by zero.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Applies Cosine similarity to inputs.

        Args:
             x1: an input1. Tensors of up to 3 dimensions
             x2: an input2. Tensors of up to 3 dimensions.

        Returns:
            cosine similarity between input1 and input2 along inserted dimension.
        """

        return function.cosine_similarity(x1, x2, self.dim, self.eps)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return f"dim={self.dim}, eps={self.eps}, backend=Trident"


class Dropout(torch.nn.Module):
    def __init__(self, p: float = 0.5):
        """
        Applies Dropout to an input.

        During training, randomly zeroes some of the elements of the input tensor with probability using samples from a
        Bernoulli distribution. Each channel will be zeroed out independently on every forward call.

        Args:
            p: probability of an element to be zeroed
        """
        super().__init__()

        self.p = p

    def forward(self, input: torch.Tensor):
        """
        Applies Dropout to an input.

        Args:
            input: an input can be of any shape

        Returns:
            an output is of the same shape as input
        """
        return function.dropout(input, self.p, self.training)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return f"p={self.p}, backend=Trident"


class GEGLU(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        use_accelerator: bool = False,
        device=None,
        dtype=None,
    ):
        """
        Applies Linear Transformation to an input.

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            bias: If set to False, the layer will not learn an additive bias
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.use_accelerator = use_accelerator
        self.reset_parameters()

    def forward(self, input: torch.Tensor):
        """
        Args:
            input: an input (*, in_features)

        Returns:
            an output (*, out_features)
        """
        return function.geglu(input, self.weight, self.bias, self.use_accelerator)

    def reset_parameters(self):
        """
        Reset parameters of the module.
        """
        bound = math.sqrt(1 / self.in_features)
        util.uniform(self.weight, -bound, bound)

        if self.bias is not None:
            util.uniform(self.bias, -bound, bound)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"backend=Trident"
        )


class GELU(torch.nn.Module):
    def __init__(self):
        """
        Applies the Gaussian Error Linear Units to an input.
        """
        super().__init__()

    def forward(self, input: torch.Tensor):
        """
        Applies the Gaussian Error Linear Units to an input.

        Args:
            input: an input can be of any shape

        Returns:
            an output is of the same shape as input
        """
        return function.gelu(input)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return f"backend=Trident"


class GroupNorm(torch.nn.Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-05,
        affine: bool = True,
        device=None,
        dtype=None,
    ):
        """
        Applies Group Normalization over a mini-batch of inputs as described in the paper Group Normalization.

        Args:
            num_groups: number of groups to separate the channels into
            num_channels: number of channels expected in input
            eps: a value added to the denominator for numerical stability
            affine: a boolean value that when set to True, this module has learnable per-channel affine
                    parameters initialized to ones (for weights) and zeros (for biases)
        """
        super().__init__()

        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = torch.nn.Parameter(torch.empty(num_channels, **factory_kwargs))
            self.bias = torch.nn.Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def forward(self, input: torch.Tensor):
        """
        Applies Group Normalization to an input.

        Args:
            input: an input

        Returns:
            an output with the same dimension and shape as an input
        """
        return function.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)

    def reset_parameters(self):
        """
        Reset parameters of the module.
        """
        if self.affine:
            util.fill(self.weight, 1)
            util.zero(self.bias)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return f"{self.num_groups}, {self.num_channels}, eps={self.eps}, affine={self.affine}, backend=Trident"


class InstanceNorm1d(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device=None,
        dtype=None,
    ):
        """
        Applies Instance Normalization to an input as described in the paper Instance Normalization: The Missing
        Ingredient for Fast Stylization.

        Args:
            num_features: C from an expected input of size (N, C, L) or (C, L)
            eps: a value added to the denominator for numerical stability
        """
        super().__init__()

        ctor_args = {"device": device, "dtype": dtype}
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(num_features, **ctor_args))
            self.bias = torch.nn.Parameter(torch.empty(num_features, **ctor_args))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.empty(num_features, **ctor_args))
            self.register_buffer("running_var", torch.empty(num_features, **ctor_args))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)

        self.reset_parameters()

    def forward(self, input: torch.Tensor):
        """
        Applies Instance Normalization to an input.

        Args:
            input: an input (N, C, L) or (C, L)

        Returns:
            an output with the same dimension and shape as an input
        """
        return function.instance_norm(
            input if input.dim() == 3 else input.view(-1, *input.shape),
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            not self.track_running_stats,
            self.momentum,
            self.eps,
        ).view(input.shape)

    def reset_parameters(self):
        """
        Reset parameters of the module.
        """
        if self.affine:
            util.fill(self.weight, 1)
            util.zero(self.bias)

        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return (
            f"{self.num_features}, "
            f"eps={self.eps}, "
            f"momentum={self.momentum}, "
            f"affine={self.affine}, "
            f"track_running_stats={self.track_running_stats}, "
            f"backend=Trident"
        )


class InstanceNorm2d(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device=None,
        dtype=None,
    ):
        """
        Applies Instance Normalization to an input as described in the paper Instance Normalization: The Missing
        Ingredient for Fast Stylization.

        Args:
            num_features: C from an expected input of size (N, C, H, W) or (C, H, W)
            eps: a value added to the denominator for numerical stability
        """
        super().__init__()

        ctor_args = {"device": device, "dtype": dtype}
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(num_features, **ctor_args))
            self.bias = torch.nn.Parameter(torch.empty(num_features, **ctor_args))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.empty(num_features, **ctor_args))
            self.register_buffer("running_var", torch.empty(num_features, **ctor_args))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)

        self.reset_parameters()

    def forward(self, input: torch.Tensor):
        """
        Applies Instance Normalization to an input.

        Args:
            input: an input (N, C, H, W) or (C, H, W)

        Returns:
            an output with the same dimension and shape as an input
        """
        if input.dim() == 4:
            inp = input.view(input.shape[0], input.shape[1], -1)
        else:
            inp = input

        out = function.instance_norm(
            inp,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            not self.track_running_stats,
            self.momentum,
            self.eps,
        )

        return out.view(input.shape)

    def reset_parameters(self):
        """
        Reset parameters of the module.
        """
        if self.affine:
            util.fill(self.weight, 1)
            util.zero(self.bias)

        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return (
            f"{self.num_features}, "
            f"eps={self.eps}, "
            f"momentum={self.momentum}, "
            f"affine={self.affine}, "
            f"track_running_stats={self.track_running_stats}, "
            f"backend=Trident"
        )


class LayerNorm(torch.nn.Module):
    def __init__(
        self,
        normalized_shape: Tuple[int, ...],
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ):
        """
        Applies Layer Normalization to an input as described in the paper Layer Normalization.

        Args:
            normalized_shape: input shape from an expected input of size
            eps: a value added to the denominator for numerical stability
            elementwise_affine: a boolean value that when set to True, this module has learnable per-element affine
                                parameters initialized to ones (for weights) and zeros (for biases)
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
            self.bias = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def forward(self, input: torch.Tensor):
        """
        Applies Layer Normalization to an input.

        Args:
            input: an input

        Returns:
            an output with the same dimension and shape as an input
        """
        return function.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    def reset_parameters(self):
        """
        Reset parameters of the module.
        """
        if self.elementwise_affine:
            util.fill(self.weight, 1)
            util.zero(self.bias)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return (
            f"{tuple(self.normalized_shape)},"
            f" eps={self.eps},"
            f" elementwise_affine={self.elementwise_affine},"
            f" backend=Trident"
        )


class LeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope: float = 1e-2):
        """
        Applies Leaky ReLU to an input.

        Args:
            negative_slope: Controls the angle of the negative slope(which is used for negative input values)
        """
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input: torch.Tensor):
        """
        Applies Leaky ReLU to an input.

        Args:
            input: an input

        Returns:
            an output with the same dimension and shape as an input
        """
        return function.leaky_relu(input, self.negative_slope)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return f"negative_slope={self.negative_slope}, backend=Trident"


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        use_accelerator: bool = False,
        device=None,
        dtype=None,
    ):
        """
        Applies Linear Transformation to an input.

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            bias: If set to False, the layer will not learn an additive bias
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.use_accelerator = use_accelerator
        self.reset_parameters()

    def forward(self, input: torch.Tensor):
        """
        Args:
            input: an input (*, in_features)

        Returns:
            an output (*, out_features)
        """
        return function.linear(input, self.weight, self.bias, self.use_accelerator)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"backend=Trident"
        )

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = util.calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            util.uniform(self.bias, -bound, bound)


class Max(torch.nn.Module):
    def __init__(self, dim: torch.int32):
        """
        Computes the max along the specified dimension in an input.

        Args:
            dim: the dimension or dimensions to reduce
        """
        super().__init__()

        self.dim = dim

    def forward(self, input: torch.Tensor):
        """
        Computes the max along the specified dimension in an input.

        Args:
            input: an input

        Returns:
            the max along the specified dimension in an input
        """
        return function.max(input, self.dim)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return f"dim={self.dim}, backend=Trident"


class Mean(torch.nn.Module):
    def __init__(self, dim: int = None):
        """
        Computes the mean along the specified dimension in an input.

        Args:
            dim: the dimension or dimensions to reduce
        """
        super().__init__()

        self.dim = dim

    def forward(self, input: torch.Tensor):
        """
        Computes the mean along the specified dimension in an input.

        Args:
            input: an input

        Returns:
            the mean along the specified dimension in an input
        """
        return function.mean(input, self.dim)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return f"dim={self.dim}, backend=Trident"


class ReLU(torch.nn.Module):
    def __init__(self):
        """
        Applies Leaky ReLU to an input.
        """
        super().__init__()

    def forward(self, input: torch.Tensor):
        """
        Applies Leaky ReLU to an input.

        Args:
            input: an input

        Returns:
            an output with the same dimension and shape as an input
        """
        return function.relu(input)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return f"backend=Trident"


class PReLU(torch.nn.Module):
    def __init__(self, num_parameters=1, init=0.25, device=None, dtype=None):
        """
        Applies PReLU to an input.

        Args:
            num_parameters: number of weight to learn
            init: the initial value of weight
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_parameters = num_parameters
        self.weight = torch.nn.Parameter(torch.empty(num_parameters, **factory_kwargs).fill_(init))

    def forward(self, input):
        """
        Applies PReLU to an input.

        Args:
            input: an input

        Returns:
            an output with the same dimension and shape as an input
        """
        return operation.PReLU.apply(input, self.weight)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return f"num_parameters={self.num_parameters}, backend=Trident"


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        normalized_shape: Tuple[int, ...],
        p: float = -1.0,
        eps: float = 1e-05,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        """
        Applies Root Mean Square Layer Normalization to an input.

        Args:
            normalized_shape: input shape from an expected input of size
            p: partial RMSNorm, valid value [0, 1] otherwise it's disabled
            eps: a value added to the denominator for numerical stability
            bias: a boolean value that when set to True, this module has learnable bias parameters.
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.normalized_shape = normalized_shape
        self.p = p
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def forward(self, input: torch.Tensor):
        """
        Applies Root Mean Square Layer Normalization to an input.

        Args:
            input: an input

        Returns:
            an output with the same dimension and shape as an input
        """
        return function.rms_norm(input, self.p, self.weight, self.bias, self.eps)

    def reset_parameters(self):
        """
        Reset parameters of the module.
        """
        util.fill(self.weight, 1.0)

        if self.bias is not None:
            util.zero(self.bias)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return (
            f"{tuple(self.normalized_shape)}, "
            f"p={self.p}, "
            f"eps={self.eps}, "
            f"bias={self.bias is not None}, "
            f"backend=Trident"
        )


class ShiftGELU(torch.nn.Module):
    def __init__(self, num_features: int, device=None, dtype=None):
        """
        Applies shift and the Gaussian Error Linear Units to an input.

        Args:
            num_features: number of features
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.bias = torch.nn.Parameter(torch.empty(num_features, **factory_kwargs))

        self.reset_parameters()

    def forward(self, input: torch.Tensor):
        """
        Applies shift and the Gaussian Error Linear Units to an input.

        Args:
            input: an input

        Returns:
            an output with the same dimension and shape as an input
        """
        return function.shift_gelu(input, self.bias)

    def reset_parameters(self):
        """
        Reset parameters of the module.
        """
        util.uniform(self.bias, 0.0, 1.0)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return f"{self.num_features}, backend=Trident"


class SiLU(torch.nn.Module):
    def __init__(self):
        """
        Applies the Sigmoid Linear Unit, element-wise to an input. The SiLU function is also known as the Swish.
        """
        super().__init__()

    def forward(self, input: torch.Tensor):
        """
        Applies the Sigmoid Linear Unit to an input.

        Args:
            input: an input

        Returns:
            an output with the same dimension and shape as an input
        """
        return function.silu(input)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return f"backend=Trident"


class Softmax(torch.nn.Module):
    def __init__(self, dim: int = None):
        """
        Applies Softmax to an input rescaling them so that an output lie in the range [0,1] and sum to 1.

        Args:
            dim: A dimension along which Softmax will be computed (so every slice along dim will sum to 1)
        """
        super().__init__()

        self.dim = dim

    def forward(self, input: torch.Tensor):
        """
        Applies Softmax to input.

        Args:
            input: an input

        Returns:
            an output with the same dimension and shape as an input with values in the range [0, 1]
        """
        return function.softmax(input, self.dim)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return f"dim={self.dim}, backend=Trident"


class Sum(torch.nn.Module):
    def __init__(self, dim: int = None):
        """
        Computes the sum along the specified dimension in an input.

        Args:
            dim: the dimension or dimensions to reduce.
        """
        super().__init__()

        self.dim = dim

    def forward(self, input: torch.Tensor):
        """
        Computes the sum along the specified dimension in an input.

        Args:
            input: an input

        Returns:
            the sum of all elements in an input
        """
        return function.sum(input, self.dim)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return f"dim={self.dim}, backend=Trident"


class Var(torch.nn.Module):
    def __init__(self, dim: int = None, correction: int = 1):
        """
        Computes the variance along the specified dimension in an input.

        Args:
            dim: the dimension or dimensions to reduce
            correction: the difference between the sample size and sample degrees of freedom
        """
        super().__init__()

        self.dim = dim
        self.correction = correction

    def forward(self, input: torch.Tensor):
        """
        Computes the variance along the specified dimension in an input.

        Args:
            input: an input

        Returns:
            the variance along the specified dimension in an input
        """
        return function.var(input, self.dim, self.correction)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return f"dim={self.dim}, correction={self.correction}, backend=Trident"


class VarMean(torch.nn.Module):
    def __init__(self, dim: int = None, correction: int = 1):
        """
        Computes the variance and mean along the specified dimension in an input.

        Args:
            dim: the dimension or dimensions to reduce
            correction: the difference between the sample size and sample degrees of freedom
        """
        super().__init__()

        self.dim = dim
        self.correction = correction

    def forward(self, input: torch.Tensor):
        """
        Computes the variance and mean along the specified dimension in an input.

        Args:
            input: an input

        Returns:
            the variance and mean along the specified dimension in an input
        """
        return function.var_mean(input, self.dim, self.correction)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return f"dim={self.dim}, correction={self.correction}, backend=Trident"
