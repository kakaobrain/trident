"""
Copyright 2023 ⓒ Kakao Brain Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import triton

import trident


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['num_elements'],
        x_vals=[2 << i for i in range(8, 18)],
        x_log=True,
        line_arg='provider',
        line_vals=['torch', 'trident'],
        line_names=['torch', 'trident'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='ms',
        plot_name='instance norm forward',
        args={'num_batches': 512, 'num_channels': 3}
    )
)
def benchmark_instance_norm_forward(num_batches, num_channels, num_elements, provider):
    x = torch.randn(num_batches, num_channels, num_elements, device='cuda')

    if provider == 'torch':
        return triton.testing.do_bench(lambda: torch.nn.functional.instance_norm(x))
    else:
        return triton.testing.do_bench(lambda: trident.function.instance_norm(x))


if __name__ == '__main__':
    benchmark_instance_norm_forward.run(print_data=True, show_plots=False)
