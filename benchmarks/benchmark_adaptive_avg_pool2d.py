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
    "adaptive avg pool2d forward",
    ["out_sz"],
    [2**i for i in range(1, 11)],
    {"h": 512, "w": 512},
)
def bench_adaptive_avg_pool2d_forward(out_sz, h, w, backend):
    inp = torch.randn(1, 1, h, w, device="cuda")

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(lambda: torch.nn.functional.adaptive_avg_pool2d(inp, out_sz))
    else:
        return triton.testing.do_bench_cudagraph(lambda: trident.function.adaptive_avg_pool2d(inp, out_sz))


def run_benchmark(mode, show_plots):
    if mode == "forward":
        bench_adaptive_avg_pool2d_forward.run(print_data=True, show_plots=show_plots)
    else:
        raise NotImplementedError("The backward isn't implemented.")
