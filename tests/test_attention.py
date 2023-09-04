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

import trident
from tests import util


@pytest.mark.parametrize("is_causal, embedding_size", [(True, 16), (False, 32), (False, 64)])
def test_forward(is_causal, embedding_size, device):
    num_batches, num_heads, sequence_size = 6, 9, 1024
    factory_kwargs = {"device": device}
    query = torch.rand(num_batches, num_heads, sequence_size, embedding_size, **factory_kwargs)
    key = torch.rand(num_batches, num_heads, sequence_size, embedding_size, **factory_kwargs)
    value = torch.rand(num_batches, num_heads, sequence_size, embedding_size, **factory_kwargs)

    a = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=is_causal)
    b = trident.function.scaled_dot_product_attention(query, key, value, is_causal=is_causal)
    assert util.equal(a, b)


@pytest.mark.parametrize("is_causal, embedding_size", [(True, 16), (False, 32), (False, 64)])
def test_backward(is_causal, embedding_size, device):
    num_batches, num_heads, sequence_size = 6, 9, 1024
    factory_kwargs = {"device": device}
    query = torch.rand(num_batches, num_heads, sequence_size, embedding_size, **factory_kwargs)
    key = torch.rand(num_batches, num_heads, sequence_size, embedding_size, **factory_kwargs)
    value = torch.rand(num_batches, num_heads, sequence_size, embedding_size, **factory_kwargs)
    grad_out = torch.rand(num_batches, num_heads, sequence_size, embedding_size, **factory_kwargs)

    def train(func):
        q = query.clone()
        k = key.clone()
        v = value.clone()
        q.requires_grad = k.requires_grad = v.requires_grad = True
        func(q, k, v, is_causal=is_causal).backward(grad_out, retain_graph=True)
        return q.grad, k.grad, v.grad

    (x, y, z) = train(torch.nn.functional.scaled_dot_product_attention)
    (a, b, c) = train(trident.function.scaled_dot_product_attention)

    assert util.equal(x, a)
    assert util.equal(y, b)
    assert util.equal(z, c)


@pytest.mark.parametrize("is_causal, embedding_size", [(True, 16)])
def test_attention(is_causal, embedding_size, device, dtype):
    num_batches, num_heads, sequence_size = 6, 9, 1024
    factory_kwargs = {"device": device, "dtype": dtype}
    query = torch.rand(num_batches, num_heads, sequence_size, embedding_size, **factory_kwargs)
    key = torch.rand(num_batches, num_heads, sequence_size, embedding_size, **factory_kwargs)
    value = torch.rand(num_batches, num_heads, sequence_size, embedding_size, **factory_kwargs)

    output = trident.function.scaled_dot_product_attention(query, key, value, is_causal=is_causal)
    assert output is not None and output.dtype == dtype
