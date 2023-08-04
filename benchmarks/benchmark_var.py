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
    "var forward",
    ["x_size"],
    [256 * i for i in range(1, 21)],
    {"y_size": 32},
)
def bench_var_forward(y_size, x_size, ctx):
    factory_kwargs = {"device": "cuda"}
    input = torch.randn(y_size, x_size, **factory_kwargs)

    if ctx == "torch":
        return triton.testing.do_bench(lambda: torch.var(input, 1))
    else:
        return triton.testing.do_bench(lambda: trident.function.var(input, 1))


@util.report(
    "var backward",
    ["x_size"],
    [256 * i for i in range(1, 21)],
    {"y_size": 32},
)
def bench_var_backward(y_size, x_size, ctx):
    factory_kwargs = {"device": "cuda", "requires_grad": True}
    input = torch.randn(y_size, x_size, **factory_kwargs)
    target = torch.empty(y_size, **factory_kwargs)

    if ctx == "torch":
        output = torch.var(input, 1)
    else:
        output = trident.function.var(input, 1)

    return triton.testing.do_bench(lambda: output.backward(target, retain_graph=True))


def run_benchmark(mode, show_plots):
    if mode == "forward":
        bench_var_forward.run(print_data=True, show_plots=show_plots)
    else:
        bench_var_backward.run(print_data=True, show_plots=show_plots)
