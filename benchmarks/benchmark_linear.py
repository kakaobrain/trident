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


@util.report("linear forward", ["m", "k", "n"], [64 * i for i in range(1, 21)])
def bench_linear_forward(m, n, k, ctx):
    inp = torch.randn(m, k, device="cuda")
    wgt = torch.randn(n, k, device="cuda")
    bis = torch.randn(n, device="cuda")

    if ctx == "torch":
        return triton.testing.do_bench(
            lambda: torch.nn.functional.linear(inp, wgt, bis)
        )
    else:
        return triton.testing.do_bench(lambda: trident.function.linear(inp, wgt, bis))


@util.report("linear backward", ["m", "k", "n"], [64 * i for i in range(1, 21)])
def bench_linear_backward(m, n, k, ctx):
    inp = torch.randn(m, k, device="cuda")

    if ctx == "torch":
        lyr = torch.nn.Linear(k, n, True, device="cuda")
    else:
        lyr = trident.Linear(k, n, True)

    out = lyr.forward(inp)
    grad_out = torch.ones_like(out)

    return triton.testing.do_bench(lambda: out.backward(grad_out, retain_graph=True))


def run_benchmark(mode, show_plots):
    if mode == "forward":
        bench_linear_forward.run(print_data=True, show_plots=show_plots)
    else:
        bench_linear_backward.run(print_data=True, show_plots=show_plots)
