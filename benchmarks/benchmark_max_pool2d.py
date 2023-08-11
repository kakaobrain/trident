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
    "max pool2d forward",
    ["knl_sz"],
    [3 * i for i in range(1, 21)],
    {"num_bt": 2, "num_ch": 3, "h": 512, "w": 512},
)
def bench_max_pool2d_forward(num_bt, num_ch, h, w, knl_sz, ctx):
    inp = torch.randn(num_bt, num_ch, h, w, device="cuda")

    if ctx == "torch":
        return triton.testing.do_bench(lambda: torch.nn.functional.max_pool2d(inp, knl_sz))
    else:
        return triton.testing.do_bench(lambda: trident.function.max_pool2d(inp, knl_sz))


def run_benchmark(mode, show_plots):
    if mode == "forward":
        bench_max_pool2d_forward.run(print_data=True, show_plots=show_plots)
    else:
        raise NotImplementedError("The backward isn't implemented.")
