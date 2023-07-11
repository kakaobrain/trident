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


@util.report("forward", "vec_sz", [256 * i for i in range(1, 21)], {"num_vec": 3})
def bench_layer_norm_forward(num_vec, vec_sz, ctx):
    inp = torch.randn(num_vec, vec_sz, device="cuda")
    norm_sh = (inp.shape[-1],)

    if ctx == "torch":
        return triton.testing.do_bench(
            lambda: torch.nn.functional.layer_norm(inp, norm_sh)
        )
    else:
        return triton.testing.do_bench(
            lambda: trident.function.layer_norm(inp, norm_sh)
        )


@util.report("backward", "vec_sz", [256 * i for i in range(1, 21)], {"num_vec": 3})
def bench_layer_norm_backward(num_vec, vec_sz, ctx):
    inp = torch.randn(num_vec, vec_sz, device="cuda", requires_grad=True)
    norm_sh = [inp.shape[-1]]

    if ctx == "torch":
        lyr = torch.nn.LayerNorm(norm_sh, dtype=torch.float32, device="cuda")
    else:
        lyr = trident.LayerNorm(norm_sh, dtype=torch.float32, device="cuda")

    out = lyr.forward(inp)
    grad_out = torch.ones_like(inp)

    return triton.testing.do_bench(lambda: out.backward(grad_out, retain_graph=True))


def run_benchmarks(mode, show_plots):
    if mode == "forward":
        bench_layer_norm_forward.run(print_data=True, show_plots=show_plots)
    elif mode == "backward":
        bench_layer_norm_backward.run(print_data=True, show_plots=show_plots)
    else:
        bench_layer_norm_forward.run(print_data=True, show_plots=show_plots)
        bench_layer_norm_backward.run(print_data=True, show_plots=show_plots)
