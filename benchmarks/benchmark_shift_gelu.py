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


def shift_gelu(input: torch.Tensor, bias: torch.Tensor):
    return torch.nn.functional.gelu(input + bias)


@util.report("shift gelu forward", ["x_size"], [512 * i for i in range(1, 21)], {"num_batches": 2048, "y_size": 16})
def bench_shift_gelu_forward(num_batches, y_size, x_size, backend):
    input = torch.randn((num_batches, y_size, x_size), device="cuda")
    bias = torch.randn(x_size, device="cuda")

    if backend == "torch":
        return triton.testing.do_bench(lambda: shift_gelu(bias, input))
    else:
        return triton.testing.do_bench(lambda: trident.function.shift_gelu(input, bias))


@util.report("shift gelu backward", ["x_size"], [512 * i for i in range(1, 21)], {"num_batches": 2048, "y_size": 16})
def bench_shift_gelu_backward(num_batches, y_size, x_size, backend):
    input = torch.randn((num_batches, y_size, x_size), device="cuda", requires_grad=True)
    bias = torch.randn(x_size, device="cuda")
    grad_output = torch.randn((num_batches, y_size, x_size), device="cuda")

    if backend == "torch":
        output = shift_gelu(input, bias)
    else:
        output = trident.function.shift_gelu(input, bias)

    return triton.testing.do_bench_cudagraph(lambda: output.backward(grad_output, retain_graph=True))


def run_benchmark(mode, show_plots):
    if mode == "forward":
        bench_shift_gelu_forward.run(print_data=True, show_plots=show_plots)
    else:
        bench_shift_gelu_backward.run(print_data=True, show_plots=show_plots)
