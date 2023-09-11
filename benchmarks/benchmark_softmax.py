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


@util.report("softmax forward", ["x_size"], [128 * i for i in range(1, 21)], {"y_size": 16})
def bench_softmax_forward(y_size, x_size, backend):
    input = torch.randn(y_size, x_size, device="cuda")

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(lambda: torch.softmax(input, 1))
    else:
        return triton.testing.do_bench_cudagraph(lambda: trident.function.softmax(input, 1))


@util.report("softmax backward", ["x_size"], [128 * i for i in range(1, 21)], {"y_size": 16})
def bench_softmax_backward(y_size, x_size, backend):
    input = torch.randn(y_size, x_size, device="cuda", requires_grad=True)
    grad_output = torch.randn(y_size, x_size, device="cuda")

    if backend == "torch":
        output = torch.softmax(input, 1)
    else:
        output = trident.function.softmax(input, 1)

    return triton.testing.do_bench_cudagraph(lambda: output.backward(grad_output, retain_graph=True))


def run_benchmark(mode, show_plots):
    if mode == "forward":
        bench_softmax_forward.run(print_data=True, show_plots=show_plots)
    else:
        bench_softmax_backward.run(print_data=True, show_plots=show_plots)
