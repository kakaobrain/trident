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

import triton


def make_benchmark(title, x_name, x_vals, args):
    return triton.testing.Benchmark(
        [x_name],
        x_vals,
        "ctx",
        ["torch", "trident"],
        ["torch", "trident"],
        title,
        args,
        ylabel="milliseconds",
        x_log=True,
    )


def report(title, x_name, x_vals, args):
    return triton.testing.perf_report(make_benchmark(title, x_name, x_vals, args))
