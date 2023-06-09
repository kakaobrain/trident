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

import triton


@triton.jit
def batch(index, num_channels, num_rows, num_cols):
    return index // (num_channels * num_rows * num_cols)


@triton.jit
def channel(index, num_channels, num_rows, num_cols):
    return (index % (num_channels * num_rows * num_cols)) // (num_rows * num_cols)


@triton.jit
def col(index, num_cols):
    return index % num_cols


@triton.jit
def max1d(x):
    return triton.language.max(x, 0)


@triton.jit
def max2d(x):
    return triton.language.max(triton.language.ravel(x), 0)


@triton.jit
def pow2(x):
    return x * x


@triton.jit
def row(index, num_rows, num_cols):
    return (index % (num_rows * num_cols)) // num_cols


@triton.jit
def var(mask, x, size, mean, dim=0, correction=1):
    return triton.language.sum(pow2(triton.language.where(mask, x - mean, 0.0)), dim) / (size - correction)


@triton.jit
def relu(x):
    return triton.language.where(x > 0, x, 0)


@triton.jit
def leaky_relu(x, a):
    return triton.language.where(x > 0, x, 0) + a * triton.language.where(x > 0, 0, x)
