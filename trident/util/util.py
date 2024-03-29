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

from typing import Callable, List

import nvtx
import torch
import triton
import triton.language as tl

from trident import config, module


def autotune(
    configs: List[triton.Config],
    key: List[str],
    prune_configs_by: Callable = None,
    reset_to_zero: List[str] = None,
    warmup: int = 25,
    rep: int = 100,
):
    return triton.autotune(
        configs if config.use_autotune else [configs[0]], key, prune_configs_by, reset_to_zero, warmup, rep
    )


def calculate_fan_in_and_fan_out(input: torch.Tensor):
    with torch.no_grad():
        return torch.nn.init._calculate_fan_in_and_fan_out(input)


def fill(inp, val):
    with torch.no_grad():
        return inp.fill_(val)


def dtype(input):
    if input == torch.float32:
        return tl.float32
    elif input == torch.float16:
        return tl.float16
    elif input == torch.bfloat16:
        return tl.bfloat16
    elif input == torch.int64:
        return tl.int64
    else:
        raise ValueError(f"Unable to convert the given input: '{input}'.")


def is_pow2(value):
    return False if value == 0 else (value & (value - 1)) == 0


def size_and_stride(input: torch.Tensor, dim: int):
    if input.dim() == 2:
        if dim == 0:
            x_size, y_size = input.shape
            y_stride = input.stride(1)
            x_stride = input.stride(0)
        else:
            y_size, x_size = input.shape
            y_stride = input.stride(0)
            x_stride = input.stride(1)

        return y_size, x_size, y_stride, x_stride
    elif input.dim() == 3:
        if dim == 0:
            z_size, y_size, x_size = input.shape[0], input.shape[1], input.shape[2]
            z_stride, y_stride, x_stride = input.stride(0), input.stride(1), input.stride(2)
        elif dim == 1:
            z_size, y_size, x_size = input.shape[1], input.shape[0], input.shape[2]
            z_stride, y_stride, x_stride = input.stride(1), input.stride(0), input.stride(2)
        else:
            z_size, y_size, x_size = input.shape[2], input.shape[0], input.shape[1]
            z_stride, y_stride, x_stride = input.stride(2), input.stride(0), input.stride(1)

        return z_size, y_size, x_size, z_stride, y_stride, x_stride
    else:
        raise ValueError(f"{dim} is not supported.")


def push_trace(message: str):
    if config.use_trace:
        nvtx.push_range(message, color="green", domain="Trident")


def pop_trace():
    if config.use_trace:
        nvtx.pop_range(domain="Trident")


def uniform(input: torch.Tensor, a: float = 0.0, b: float = 1.0):
    with torch.no_grad():
        return torch.nn.init.uniform(input, a, b)


def zero(inp):
    with torch.no_grad():
        return inp.zero_()
