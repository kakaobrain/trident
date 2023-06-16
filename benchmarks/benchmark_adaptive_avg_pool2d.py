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
        x_names=['output_size'],
        x_vals=[2 << i for i in range(0, 8)],
        line_arg='provider',
        line_vals=['torch', 'trident'],
        line_names=['torch', 'trident'],
        plot_name='adaptive avg pool2d forward',
        args={'num_rows': 512, 'num_cols': 512},
        ylabel='milliseconds',
        x_log=True
    )
)
def bench_adaptive_avg_pool2d_forward(output_size, num_rows, num_cols, provider):
    x = torch.randn(1, 1, num_rows, num_cols, device='cuda')

    if provider == 'torch':
        return triton.testing.do_bench(lambda: torch.nn.functional.adaptive_avg_pool2d(x, output_size))
    else:
        return triton.testing.do_bench(lambda: trident.function.adaptive_avg_pool2d(x, output_size))


def run_benchmarks(show_plots):
    bench_adaptive_avg_pool2d_forward.run(print_data=True, show_plots=show_plots)