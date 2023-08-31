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


@util.report("layer norm forward", ["x_size"], [128 * i for i in range(1, 21)], {"num_batches": 4, "y_size": 2048})
def bench_layer_norm_forward(num_batches, y_size, x_size, backend):
    input = torch.randn((num_batches, y_size, x_size), device="cuda")
    normalized_shape = (input.shape[-1],)
    weight = torch.randn(x_size, device="cuda")
    bias = torch.randn(x_size, device="cuda")

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(
            lambda: torch.nn.functional.layer_norm(input, normalized_shape, weight, bias)
        )
    else:
        return triton.testing.do_bench_cudagraph(
            lambda: trident.function.layer_norm(input, normalized_shape, weight, bias)
        )


@util.report("layer norm backward", ["x_size"], [128 * i for i in range(1, 21)], {"num_batches": 512, "y_size": 2048})
def bench_layer_norm_backward(num_batches, y_size, x_size, backend):
    input = torch.randn((num_batches, y_size, x_size), device="cuda", requires_grad=True)
    weight = torch.randn(x_size, device="cuda", requires_grad=True)
    bias = torch.randn(x_size, device="cuda", requires_grad=True)
    normalized_shape = (input.shape[-1],)
    grad_output = torch.randn((num_batches, y_size, x_size), device="cuda")

    if backend == "torch":
        output = torch.nn.functional.layer_norm(input, normalized_shape, weight, bias)
    else:
        output = trident.function.layer_norm(input, normalized_shape, weight, bias)

    return triton.testing.do_bench_cudagraph(lambda: output.backward(grad_output, retain_graph=True))


def run_benchmark(mode, show_plots):
    if mode == "forward":
        bench_layer_norm_forward.run(print_data=True, show_plots=show_plots)
    else:
        bench_layer_norm_backward.run(print_data=True, show_plots=show_plots)
