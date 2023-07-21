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
    ["vec_sz"],
    [256 * i for i in range(1, 21)],
    {"num_vec": 3, "num_grp": 8},
)
def bench_group_norm_forward(num_vec, num_grp, vec_sz, ctx):
    inp = torch.randn(num_vec, vec_sz, device="cuda")

    if ctx == "torch":
        return triton.testing.do_bench(
            lambda: torch.nn.functional.group_norm(inp, num_grp)
        )
    else:
        return triton.testing.do_bench(
            lambda: trident.function.group_norm(inp, num_grp)
        )


@util.report(
    "group norm backward",
    ["vec_sz"],
    [256 * i for i in range(1, 21)],
    {"num_vec": 3, "num_grp": 8},
)
def bench_group_norm_backward(num_vec, num_grp, vec_sz, ctx):
    inp = torch.randn(num_vec, vec_sz, device="cuda", requires_grad=True)

    if ctx == "torch":
        lyr = torch.nn.GroupNorm(num_grp, vec_sz, dtype=torch.float32, device="cuda")
    else:
        lyr = trident.GroupNorm(num_grp, vec_sz, dtype=torch.float32, device="cuda")

    out = lyr.forward(inp)
    grad_out = torch.ones_like(inp)

    return triton.testing.do_bench(lambda: out.backward(grad_out, retain_graph=True))


def run_benchmark(mode, show_plots):
    if mode == "forward":
        bench_group_norm_forward.run(print_data=True, show_plots=show_plots)
    else:
        bench_group_norm_backward.run(print_data=True, show_plots=show_plots)
