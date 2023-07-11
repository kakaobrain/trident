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
    "conv2d forward",
    "wgt_sz",
    [3 * i for i in range(1, 21)],
    {"num_bt": 2, "inp_ch": 3, "inp_sz": 256, "out_ch": 8},
)
def bench_conv2d_forward(num_bt, inp_ch, inp_sz, out_ch, wgt_sz, ctx):
    inp = torch.randn(num_bt, inp_ch, inp_sz, inp_sz, device="cuda")
    wgt = torch.randn(out_ch, inp_ch, wgt_sz, wgt_sz, device="cuda")

    if ctx == "torch":
        return triton.testing.do_bench(lambda: torch.nn.functional.conv2d(inp, wgt))
    else:
        return triton.testing.do_bench(lambda: trident.function.conv2d(inp, wgt))


def run_benchmarks(mode, show_plots):
    if mode == "forward":
        bench_conv2d_forward.run(print_data=True, show_plots=show_plots)
    elif mode == "backward":
        pass
    else:
        bench_conv2d_forward.run(print_data=True, show_plots=show_plots)
