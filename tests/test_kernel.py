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

import pytest
import torch
import triton

import trident
from tests import util


@pytest.mark.parametrize("num_vec, vec_sz, axis", [(10, 1024, 0), (5, 10, 1)])
def test_sum(num_vec, vec_sz, axis, device, dtype):
    ctor_args = {"device": device, "dtype": dtype}
    inp = torch.randn(num_vec, vec_sz, **ctor_args)
    out_sz = vec_sz if axis == 0 else num_vec
    out = torch.empty(out_sz, **ctor_args)

    def grid(meta):
        return [out_sz]

    trident.kernel.sum[grid](
        out,
        inp,
        num_vec,
        vec_sz,
        axis,
        trident.util.block_size(num_vec if axis == 0 else vec_sz, inp.element_size()),
    )

    assert util.equal(torch.sum(inp, dim=axis), out)
