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


def geglu(input, weight, bias: torch.Tensor = None):
    hidden_state, gate = torch.nn.functional.linear(input, weight, bias).chunk(2, -1)
    return hidden_state * torch.nn.functional.gelu(gate)


@util.report("geglu forward", ["m_size", "n_size", "k_size"], [128 * i for i in range(1, 11)], {"num_batches": 16})
def bench_geglu_forward(num_batches, m_size, n_size, k_size, backend):
    input = torch.randn(num_batches, m_size, k_size, device="cuda")
    weight = torch.randn(n_size, k_size, device="cuda")
    bias = torch.randn(n_size, device="cuda")

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(lambda: geglu(input, weight, bias))
    else:
        return triton.testing.do_bench_cudagraph(
            lambda: trident.function.geglu(input, weight, bias, use_accelerator=True)
        )


@util.report("linear backward", ["m_size", "n_size", "k_size"], [128 * i for i in range(1, 11)], {"num_batches": 16})
def bench_geglu_backward(num_batches, m_size, n_size, k_size, backend):
    input = torch.randn(num_batches, m_size, k_size, device="cuda", requires_grad=True)
    weight = torch.randn(n_size, k_size, device="cuda", requires_grad=True)
    bias = torch.randn(n_size, device="cuda", requires_grad=True)
    grad_output = torch.randn(num_batches, m_size, n_size // 2, device="cuda")

    if backend == "torch":
        output = geglu(input, weight, bias)
    else:
        output = trident.function.geglu(input, weight, bias, use_accelerator=True)

    return triton.testing.do_bench_cudagraph(lambda: output.backward(grad_output, retain_graph=True))


def run_benchmark(mode, show_plots):
    if mode == "forward":
        bench_geglu_forward.run(print_data=True, show_plots=show_plots)
    else:
        bench_geglu_backward.run(print_data=True, show_plots=show_plots)
