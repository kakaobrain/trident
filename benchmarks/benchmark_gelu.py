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


@util.report("gelu forward", ["x_size"], [256 * i for i in range(1, 21)], {"num_batches": 64})
def bench_gelu_forward(num_batches, x_size, backend):
    input = torch.randn((num_batches, x_size), device="cuda")

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(lambda: torch.nn.functional.gelu(input))
    else:
        return triton.testing.do_bench_cudagraph(lambda: trident.function.gelu(input))


@util.report("gelu forward", ["x_size"], [256 * i for i in range(1, 21)], {"num_batches": 64})
def bench_gelu_backward(num_batches, x_size, backend):
    input = torch.randn((num_batches, x_size), device="cuda", requires_grad=True)

    if backend == "torch":
        output = torch.nn.functional.gelu(input)
    else:
        output = trident.function.gelu(input)

    target = torch.randn((num_batches, x_size), device="cuda")

    return triton.testing.do_bench_cudagraph(lambda: output.backward(target, retain_graph=True))


def run_benchmark(mode, show_plots):
    if mode == "forward":
        bench_gelu_forward.run(print_data=True, show_plots=show_plots)
    else:
        bench_gelu_backward.run(print_data=True, show_plots=show_plots)
