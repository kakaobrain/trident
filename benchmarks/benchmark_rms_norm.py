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
import util

import trident


def rms_norm(input: torch.Tensor, p: float, weight: torch.Tensor, bias: torch.Tensor = None, eps: float = 1e-08):
    y_size, x_size = input.shape

    if p < 0.0 or p > 1.0:
        norm = input.norm(2, dim=-1, keepdim=True)
        partial_size = x_size
    else:
        partial_size = int(x_size * p)
        partial_input, _ = torch.split(input, [partial_size, x_size - partial_size], dim=-1)
        norm = partial_input.norm(2, dim=-1, keepdim=True)

    rms = norm * partial_size ** (-1.0 / 2)
    output = input / (rms + eps)

    if bias:
        return weight * output + bias

    return weight * output


@util.report(
    "rms norm forward", ["num_batches"], [8 * i for i in range(1, 21)], {"y_size": 2048, "x_size": 2048, "p": 1.0}
)
def bench_rms_norm_forward(num_batches, y_size, x_size, p, dtype, backend):
    factory_kwargs = {"device": "cuda", "dtype": dtype}
    input = torch.randn(num_batches * y_size, x_size, **factory_kwargs)
    weight = torch.randn(x_size, **factory_kwargs)

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(lambda: rms_norm(input, p, weight))
    else:
        return triton.testing.do_bench_cudagraph(lambda: trident.function.rms_norm(input, p, weight))


@util.report(
    "rms norm backward", ["num_batches"], [8 * i for i in range(1, 21)], {"y_size": 2048, "x_size": 2048, "p": 1.0}
)
def bench_rms_norm_backward(num_batches, y_size, x_size, p, dtype, backend):
    factory_kwargs = {"device": "cuda", "dtype": dtype}
    input = torch.randn(num_batches * y_size, x_size, **factory_kwargs, requires_grad=True)
    weight = torch.randn(x_size, **factory_kwargs, requires_grad=True)
    grad_output = torch.randn(num_batches * y_size, x_size, **factory_kwargs)

    if backend == "torch":
        output = rms_norm(input, p, weight)
    else:
        output = trident.function.rms_norm(input, p, weight)

    return triton.testing.do_bench_cudagraph(lambda: output.backward(grad_output, retain_graph=True))


def run_benchmark(mode, show_plots, dtype):
    if mode == "forward":
        bench_rms_norm_forward.run(print_data=True, show_plots=show_plots, dtype=dtype)
    else:
        bench_rms_norm_backward.run(print_data=True, show_plots=show_plots, dtype=dtype)
