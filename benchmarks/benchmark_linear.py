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
import torch.nn.functional as tf
import triton
import triton.testing as tt
import trident


@tt.perf_report(
    tt.Benchmark(
        x_names=['k'],
        x_vals=[512 * i for i in range(1, 20)],
        x_log=True,
        line_arg='provider',
        line_vals=['torch', 'trident'],
        line_names=['torch', 'trident'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='TFLOPS',
        plot_name='linear-performance',
        args={}
    )
)
def benchmark(k, provider):
    m = 1024
    n = 1024
    x = torch.randn(m, k, device='cuda', dtype=torch.float32)
    w = torch.randn(n, k, device='cuda')
    b = torch.randn(n, device='cuda')

    if provider == 'torch':
        avg_ms, min_ms, max_ms = triton.testing.do_bench(lambda: tf.linear(x, w, b))
    else:
        avg_ms, min_ms, max_ms = triton.testing.do_bench(lambda: trident.function.linear(x, w, b))

    gbps = lambda ms: (m * k * n * 2 * 1e-12) / (ms * 1e-3)
    return gbps(avg_ms), gbps(max_ms), gbps(min_ms)


if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=False)
