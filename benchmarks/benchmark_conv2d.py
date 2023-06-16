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
        x_names=['kernel_size'],
        x_vals=[i for i in range(2, 23, 2)],
        line_arg='provider',
        line_vals=['torch', 'trident'],
        line_names=['torch', 'trident'],
        plot_name='conv2d forward',
        args={'num_batches': 1, 'in_channels': 3, 'in_size': 64, 'out_channels': 16},
        ylabel='milliseconds',
        x_log=True
    )
)
def bench_conv2d_forward(num_batches, in_channels, in_size, out_channels, kernel_size, provider):
    input = torch.randn(num_batches, in_channels, in_size, in_size, dtype=torch.float, device='cuda')
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float, device='cuda')

    if provider == 'torch':
        return triton.testing.do_bench(lambda: torch.nn.functional.conv2d(input, weight))
    else:
        return triton.testing.do_bench(lambda: trident.function.conv2d(input, weight))


def run_benchmarks(show_plots):
    bench_conv2d_forward.run(print_data=True, show_plots=show_plots)
