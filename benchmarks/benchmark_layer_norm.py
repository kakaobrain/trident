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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['vec_sz'],
        x_vals=[512 * i for i in range(1, 21)],
        line_arg='provider',
        line_vals=['torch', 'trident'],
        line_names=['torch', 'trident'],
        plot_name='layer norm',
        args={'num_vec': 3},
        ylabel='milliseconds',
        x_log=True
    )
)
def bench_layer_norm_forward(num_vec, vec_sz, provider):
    inp = torch.randn(num_vec, vec_sz, device='cuda')
    norm_sh = (inp.shape[-1],)

    if provider == 'torch':
        return triton.testing.do_bench(lambda: torch.nn.functional.layer_norm(inp, norm_sh))
    else:
        return triton.testing.do_bench(lambda: trident.function.layer_norm(inp, norm_sh))


def run_benchmarks(show_plots):
    bench_layer_norm_forward.run(print_data=True, show_plots=show_plots)
