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
import torch.nn.functional as tf
import triton
import triton.testing as tt

import trident


@tt.perf_report(
    tt.Benchmark(
        x_names=['num_elements'],
        x_vals=[2 << i for i in range(4, 14)],
        line_arg='provider',
        line_vals=['torch', 'trident'],
        line_names=['torch', 'trident'],
        plot_name='linear forward',
        args={'in_features': 32, 'out_features': 32},
        ylabel='milliseconds',
        x_log=True
    )
)
def bench_linear_forward(in_features, out_features, num_elements, provider):
    x = torch.randn(in_features, num_elements, device='cuda')
    w = torch.randn(out_features, num_elements, device='cuda')
    b = torch.randn(out_features, device='cuda')

    if provider == 'torch':
        return triton.testing.do_bench(lambda: tf.linear(x, w, b))
    else:
        return triton.testing.do_bench(lambda: trident.function.linear(x, w, b))


def run_benchmarks(show_plots):
    bench_linear_forward.run(print_data=True, show_plots=show_plots)
