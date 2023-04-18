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

import collections
import torch
import triton
import trident

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['n'],
        x_vals=[128 * i for i in range(1, 21)],
        x_log=True,
        line_arg='provider',
        line_vals=['torch forward', 'trident forward', 'torch backward', 'trident backward'],
        line_names=['torch forward', 'trident forward', 'torch backward', 'trident backward'],
        styles=[('blue', '-'), ('green', '-'), ('blue', '--'), ('green', '--')],
        ylabel='TFLOPS',
        plot_name='softmax-performance',
        args={'m' : 2048}
    )
)
def benchmark(m, n, provider):
    Context = collections.namedtuple('Context', ['saved_tensors'])

    x = torch.randn(m, n, requires_grad=True, device='cuda')
    y = torch.softmax(x, 1)
    t = torch.randn(m, n, device='cuda')

    if provider == 'torch forward':
        avg_ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, 1))
    elif provider == 'trident forward':
        avg_ms, min_ms, max_ms = triton.testing.do_bench(lambda: trident.operation.Softmax.apply(x, 1))
    elif provider == 'torch backward':
        avg_ms, min_ms, max_ms = triton.testing.do_bench(lambda: y - t)
    else:
        avg_ms, min_ms, max_ms = triton.testing.do_bench(lambda: trident.operation.Softmax.backward(Context([y]), t))

    gbps = lambda ms: (2 * x.nelement() * x.element_size() * 1e-12) / (ms * 1e-3)
    return gbps(avg_ms), gbps(max_ms), gbps(min_ms)


if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=False)
