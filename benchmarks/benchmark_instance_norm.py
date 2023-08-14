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
    "instance norm forward",
    ["vec_sz"],
    [256 * i for i in range(1, 21)],
    {"num_bt": 32, "num_ch": 64},
)
def bench_instance_norm_forward(num_bt, num_ch, vec_sz, backend):
    inp = torch.randn(num_bt, num_ch, vec_sz, device="cuda")

    if backend == "torch":
        return triton.testing.do_bench_cudagraph(lambda: torch.nn.functional.instance_norm(inp))
    else:
        return triton.testing.do_bench_cudagraph(lambda: trident.function.instance_norm(inp))


def run_benchmark(mode, show_plots):
    if mode == "forward":
        bench_instance_norm_forward.run(print_data=True, show_plots=show_plots)
    else:
        raise NotImplementedError("The backward isn't implemented.")
