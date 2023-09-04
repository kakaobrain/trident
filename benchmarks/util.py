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
import triton


def dtype(input: str):
    if input == "float32":
        return torch.float32
    elif input == "float16":
        return torch.float16
    elif input == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unable to convert the given input: '{input}'.")


def make_benchmark(title, x_names, x_vals, args):
    return triton.testing.Benchmark(
        x_names,
        x_vals,
        "backend",
        ["torch", "trident"],
        ["torch", "trident"],
        title,
        args,
        ylabel="milliseconds",
    )


def report(title, x_names, x_vals, args=None):
    if args is None:
        args = {}
    return triton.testing.perf_report([make_benchmark(title, x_names, x_vals, args)])
