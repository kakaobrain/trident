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


@pytest.mark.parametrize(
    "num_batches, num_heads, y_size, x_size, is_causal", [(4, 8, 128, 64, True), (4, 8, 128, 64, False)]
)
def test_forward(num_batches, num_heads, y_size, x_size, is_causal, device):
    query = torch.randn(num_batches, num_heads, y_size, x_size, device=device)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    assert util.equal(
        torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=is_causal),
        trident.function.scaled_dot_product_attention(query, key, value, is_causal=is_causal),
    )


@pytest.mark.parametrize(
    "num_batches, num_heads, y_size, x_size, is_causal", [(4, 8, 128, 64, True), (4, 8, 128, 64, False)]
)
def test_backward(num_batches, num_heads, y_size, x_size, is_causal, device):
    query = torch.rand(num_batches, num_heads, y_size, x_size, device=device)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    grad_output = torch.randn_like(query)

    def train(func):
        i = query.clone()
        j = key.clone()
        k = value.clone()
        i.requires_grad = j.requires_grad = k.requires_grad = True
        func(i, j, k, is_causal=is_causal).backward(grad_output, retain_graph=True)
        return i.grad, j.grad, k.grad

    (x, y, z) = train(torch.nn.functional.scaled_dot_product_attention)
    (a, b, c) = train(trident.function.scaled_dot_product_attention)

    assert util.equal(x, a)
    assert util.equal(y, b)
    assert util.equal(z, c)


@pytest.mark.parametrize(
    "num_batches, num_heads, y_size, x_size, is_causal", [(1, 1, 1, 16, True), (1, 1, 1, 16, False)]
)
def test_attention(num_batches, num_heads, y_size, x_size, is_causal, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    query = torch.rand(num_batches, num_heads, y_size, x_size, **factory_kwargs, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    grad_output = torch.randn_like(query)

    output = trident.function.scaled_dot_product_attention(query, key, value, is_causal=is_causal)

    assert output is not None
    assert output.dtype == dtype

    output.backward(grad_output)

    assert query.grad is not None
    assert query.grad.dtype == dtype
    assert key.grad is not None
    assert key.grad.dtype == dtype
    assert value.grad is not None
    assert value.grad.dtype == dtype
