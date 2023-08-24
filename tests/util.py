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


def equal(a, b):
    return torch.allclose(a, b, rtol=0, atol=1e-02)


def train(inp, tgt, mod, act=None, crit=torch.nn.MSELoss()):
    out = mod(inp)
    if act is not None:
        out = activate(out, act)

    out.retain_grad()
    crit(out, tgt).backward()

    return out


def activate(inp, act):
    if act == "relu":
        return torch.relu(inp)
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu(inp)
    else:
        raise ValueError(f"{act} is not supported.")
