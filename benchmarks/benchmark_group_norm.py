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


@util.report(
    "group norm forward",
    ["y_size"],
    [256 * i for i in range(1, 11)],
    {"num_batches": 2, "x_size": 64, "num_groups": 32},
)
def bench_group_norm_forward(num_batches, y_size, x_size, num_groups, dtype, backend):
    input = torch.randn((num_batches, y_size, x_size), device="cuda", dtype=dtype)

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(lambda: torch.nn.functional.group_norm(input, num_groups))
    else:
        return triton.testing.do_bench_cudagraph(lambda: trident.function.group_norm(input, num_groups))


@util.report(
    "group norm backward",
    ["y_size"],
    [256 * i for i in range(1, 11)],
    {"num_batches": 2, "x_size": 64, "num_groups": 32},
)
def bench_group_norm_backward(num_batches, y_size, x_size, num_groups, dtype, backend):
    factory_kwargs = {"device": "cuda", "dtype": dtype}
    input = torch.randn((num_batches, y_size, x_size), **factory_kwargs, requires_grad=True)
    grad_output = torch.randn((num_batches, y_size, x_size), **factory_kwargs)

    if backend == "torch":
        output = torch.group_norm(input, num_groups)
    else:
        output = trident.function.group_norm(input, num_groups)

    return triton.testing.do_bench_cudagraph(lambda: output.backward(grad_output, retain_graph=True))


def run_benchmark(mode, show_plots, dtype):
    if mode == "forward":
        bench_group_norm_forward.run(print_data=True, show_plots=show_plots, dtype=dtype)
    else:
        bench_group_norm_backward.run(print_data=True, show_plots=show_plots, dtype=dtype)
