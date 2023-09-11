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


@util.report("prelu forward", ["y_size"], [256 * i for i in range(1, 11)], {"num_batches": 64, "x_size": 512})
def bench_prelu_forward(num_batches, y_size, x_size, backend):
    input = torch.randn(num_batches, y_size, x_size, device="cuda")
    weight = torch.randn(y_size, device="cuda")

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(lambda: torch.nn.functional.prelu(input, weight))
    else:
        return triton.testing.do_bench_cudagraph(lambda: trident.function.prelu(input, weight))


@util.report("prelu backward", ["y_size"], [256 * i for i in range(1, 11)], {"num_batches": 64, "x_size": 512})
def bench_prelu_backward(num_batches, y_size, x_size, backend):
    input = torch.randn(num_batches, y_size, x_size, device="cuda", requires_grad=True)
    weight = torch.randn(y_size, device="cuda", requires_grad=True)
    grad_output = torch.rand_like(input)

    if backend == "torch":
        output = torch.nn.functional.prelu(input, weight)
    else:
        output = trident.function.prelu(input, weight)

    return triton.testing.do_bench_cudagraph(lambda: output.backward(grad_output, retain_graph=True))


def run_benchmark(mode, show_plots):
    if mode == "forward":
        bench_prelu_forward.run(print_data=True, show_plots=show_plots)
    else:
        bench_prelu_backward.run(print_data=True, show_plots=show_plots)
