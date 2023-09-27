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

import argparse

import benchmark_argmax
import benchmark_attention
import benchmark_batch_norm
import benchmark_cosine_similarity
import benchmark_dropout
import benchmark_geglu
import benchmark_gelu
import benchmark_group_norm
import benchmark_instance_norm
import benchmark_layer_norm
import benchmark_leaky_relu
import benchmark_linear
import benchmark_masked_softmax
import benchmark_max
import benchmark_mean
import benchmark_prelu
import benchmark_relu
import benchmark_rms_norm
import benchmark_shift_gelu
import benchmark_silu
import benchmark_softmax
import benchmark_sum
import benchmark_var
import benchmark_var_mean
import torch
import util


def print_scenarios():
    print(f"Following scenarios can be chosen:")
    print(
        ", ".join(
            [
                "argmax",
                "attention",
                "batch-norm",
                "cosine-similarity",
                "dropout",
                "geglu",
                "gelu",
                "group-norm",
                "instance-norm",
                "layer-norm",
                "leaky-relu",
                "linear",
                "masked-softmax",
                "max",
                "mean",
                "prelu",
                "relu",
                "rms-norm",
                "shift-gelu",
                "silu",
                "softmax",
                "sum",
                "var",
                "var_mean",
            ]
        )
    )


def run_benchmarks(scenario, mode, show_plots, dtype):
    if scenario == "argmax":
        benchmark_argmax.run_benchmark(mode, show_plots, dtype)
    elif scenario == "attention":
        benchmark_attention.run_benchmark(mode, show_plots, dtype)
    elif scenario == "batch-norm":
        benchmark_batch_norm.run_benchmark(mode, show_plots, dtype)
    elif scenario == "cosine-similarity":
        benchmark_cosine_similarity.run_benchmark(mode, show_plots, dtype)
    elif scenario == "dropout":
        benchmark_dropout.run_benchmark(mode, show_plots, dtype)
    elif scenario == "geglu":
        benchmark_geglu.run_benchmark(mode, show_plots, dtype)
    elif scenario == "gelu":
        benchmark_gelu.run_benchmark(mode, show_plots, dtype)
    elif scenario == "group-norm":
        benchmark_group_norm.run_benchmark(mode, show_plots, dtype)
    elif scenario == "instance-norm":
        benchmark_instance_norm.run_benchmark(mode, show_plots, dtype)
    elif scenario == "layer-norm":
        benchmark_layer_norm.run_benchmark(mode, show_plots, dtype)
    elif scenario == "leaky-relu":
        benchmark_leaky_relu.run_benchmark(mode, show_plots, dtype)
    elif scenario == "linear":
        benchmark_linear.run_benchmark(mode, show_plots, dtype)
    elif scenario == "masked-softmax":
        benchmark_masked_softmax.run_benchmark(mode, show_plots, dtype)
    elif scenario == "max":
        benchmark_max.run_benchmark(mode, show_plots, dtype)
    elif scenario == "mean":
        benchmark_mean.run_benchmark(mode, show_plots, dtype)
    elif scenario == "prelu":
        benchmark_prelu.run_benchmark(mode, show_plots, dtype)
    elif scenario == "relu":
        benchmark_relu.run_benchmark(mode, show_plots, dtype)
    elif scenario == "rms-norm":
        benchmark_rms_norm.run_benchmark(mode, show_plots, dtype)
    elif scenario == "shift-gelu":
        benchmark_shift_gelu.run_benchmark(mode, show_plots, dtype)
    elif scenario == "silu":
        benchmark_silu.run_benchmark(mode, show_plots, dtype)
    elif scenario == "softmax":
        benchmark_softmax.run_benchmark(mode, show_plots, dtype)
    elif scenario == "sum":
        benchmark_sum.run_benchmark(mode, show_plots, dtype)
    elif scenario == "var":
        benchmark_var.run_benchmark(mode, show_plots, dtype)
    elif scenario == "var-mean":
        benchmark_var_mean.run_benchmark(mode, show_plots, dtype)
    elif not scenario:
        benchmark_argmax.run_benchmark(mode, show_plots, dtype)
        benchmark_attention.run_benchmark(mode, show_plots, dtype)
        benchmark_batch_norm.run_benchmark(mode, show_plots, dtype)
        benchmark_cosine_similarity.run_benchmark(mode, show_plots, dtype)
        benchmark_dropout.run_benchmark(mode, show_plots, dtype)
        benchmark_geglu.run_benchmark(mode, show_plots, dtype)
        benchmark_gelu.run_benchmark(mode, show_plots, dtype)
        benchmark_group_norm.run_benchmark(mode, show_plots, dtype)
        benchmark_instance_norm.run_benchmark(mode, show_plots, dtype)
        benchmark_layer_norm.run_benchmark(mode, show_plots, dtype)
        benchmark_leaky_relu.run_benchmark(mode, show_plots, dtype)
        benchmark_linear.run_benchmark(mode, show_plots, dtype)
        benchmark_masked_softmax.run_benchmark(mode, show_plots, dtype)
        benchmark_max.run_benchmark(mode, show_plots, dtype)
        benchmark_mean.run_benchmark(mode, show_plots, dtype)
        benchmark_prelu.run_benchmark(mode, show_plots, dtype)
        benchmark_relu.run_benchmark(mode, show_plots, dtype)
        benchmark_rms_norm.run_benchmark(mode, show_plots, dtype)
        benchmark_shift_gelu.run_benchmark(mode, show_plots, dtype)
        benchmark_silu.run_benchmark(mode, show_plots, dtype)
        benchmark_softmax.run_benchmark(mode, show_plots, dtype)
        benchmark_sum.run_benchmark(mode, show_plots, dtype)
        benchmark_var.run_benchmark(mode, show_plots, dtype)
        benchmark_var_mean.run_benchmark(mode, show_plots, dtype)
    else:
        print_scenarios()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", help="specify a scenario to run", type=str)
    parser.add_argument(
        "--mode",
        choices=["forward", "backward"],
        default="forward",
        help="specify a mode to run",
        type=str,
    )
    parser.add_argument("--show-plots", action="store_true", help="show plots")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="specify a dtype to run",
        type=str,
    )
    parser.add_argument("--list", action="store_true", help="list all scenarios can be run")
    args = parser.parse_args()

    if args.list:
        print_scenarios()
    else:
        torch.cuda.set_stream(torch.cuda.Stream())
        run_benchmarks(
            args.scenario.replace("_", "-") if args.scenario else None,
            args.mode,
            args.show_plots,
            util.dtype(args.dtype),
        )


if __name__ == "__main__":
    main()
