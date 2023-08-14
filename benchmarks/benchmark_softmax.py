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
    "softmax forward",
    ["vec_sz"],
    [256 * i for i in range(1, 21)],
    {"num_bt": 32},
)
def bench_softmax_forward(num_bt, vec_sz, backend):
    inp = torch.randn(num_bt, vec_sz, device="cuda")

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(lambda: torch.softmax(inp, 1))
    else:
        return triton.testing.do_bench_cudagraph(lambda: trident.function.softmax(inp, 1))


@util.report(
    "softmax backward",
    ["vec_sz"],
    [256 * i for i in range(1, 21)],
    {"num_bt": 32},
)
def bench_softmax_backward(num_bt, vec_sz, backend):
    inp = torch.randn(num_bt, vec_sz, device="cuda", requires_grad=True)

    if backend == "torch":
        lyr = torch.nn.Softmax(1)
    else:
        lyr = trident.Softmax(1)

    out = lyr.forward(inp)
    grad_out = torch.ones_like(inp)

    return triton.testing.do_bench_cudagraph(lambda: out.backward(grad_out, retain_graph=True))


def run_benchmark(mode, show_plots):
    if mode == "forward":
        bench_softmax_forward.run(print_data=True, show_plots=show_plots)
    else:
        raise NotImplementedError("The backward isn't implemented.")
