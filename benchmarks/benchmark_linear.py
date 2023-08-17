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


@util.report("linear forward", ["m_size", "n_size", "k_size"], [64 * i for i in range(1, 21)])
def bench_linear_forward(m_size, n_size, k_size, backend):
    input = torch.randn(m_size, k_size, device="cuda")
    weight = torch.randn(n_size, k_size, device="cuda")
    bias = torch.randn(n_size, device="cuda")

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(lambda: torch.nn.functional.linear(input, weight, bias))
    else:
        return triton.testing.do_bench_cudagraph(
            lambda: trident.function.linear(input, weight, bias, use_accelerator=True)
        )


@util.report("linear backward", ["m_size", "n_size", "k_size"], [64 * i for i in range(1, 21)])
def bench_linear_backward(m_size, n_size, k_size, backend):
    input = torch.randn(m_size, k_size, device="cuda")

    if backend == "torch":
        operation = torch.nn.Linear(k_size, n_size, True, device="cuda")
    else:
        operation = trident.Linear(k_size, n_size, True)

    output = operation.forward(input)
    grad_output = torch.randn(output)

    return triton.testing.do_bench_cudagraph(lambda: output.backward(grad_output, retain_graph=True))


def run_benchmark(mode, show_plots):
    if mode == "forward":
        bench_linear_forward.run(print_data=True, show_plots=show_plots)
    else:
        bench_linear_backward.run(print_data=True, show_plots=show_plots)
