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

import trident
import util


@util.report('linear forward', 'k', [256 * i for i in range(1, 21)], {'m': 32, 'n': 32})
def bench_linear_forward(m, n, k, ctx):
    inp = torch.randn(m, k, device='cuda')
    wgt = torch.randn(n, k, device='cuda')
    bis = torch.randn(n, device='cuda')

    if ctx == 'torch':
        return triton.testing.do_bench(lambda: torch.nn.functional.linear(inp, wgt, bis))
    else:
        return triton.testing.do_bench(lambda: trident.function.linear(inp, wgt, bis))


def run_benchmarks(mode, show_plots):
    if mode == 'forward':
        bench_linear_forward.run(print_data=True, show_plots=show_plots)
    elif mode == 'backward':
        pass
    else:
        bench_linear_forward.run(print_data=True, show_plots=show_plots)
