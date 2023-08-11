import pytest
import torch

import trident
from tests import util


@pytest.mark.parametrize(
    "num_batches, y_size, x_size, dim",
    [(5431, 500, 200, 0), (221, 1250, 200, 1), (21, 6400, 86, 2)],
)
def test_forward(num_batches, y_size, x_size, dim, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}

    input = torch.randn(num_batches, y_size, x_size, **factory_kwargs)
    other = torch.randn(num_batches, y_size, x_size, **factory_kwargs)

    assert util.equal(
        torch.nn.functional.cosine_similarity(input, other, dim=dim),
        trident.function.cosine_similarity(input, other, dim=dim),
    )


@pytest.mark.parametrize(
    "num_batches, y_size, x_size, dim",
    [(1280, 1000, 200, 0), (200, 1280, 200, 1), (640, 21, 86, 2)],
)
def test_backward(num_batches, y_size, x_size, dim, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}

    input = torch.randn(num_batches, y_size, x_size, **factory_kwargs)
    other = torch.randn(num_batches, y_size, x_size, **factory_kwargs)

    if dim == 0:
        target_dim = (y_size, x_size)
    elif dim == 1:
        target_dim = (num_batches, x_size)
    else:
        target_dim = (num_batches, y_size)

    target = torch.randn(target_dim, **factory_kwargs)

    def train(func):
        x1 = input.clone()
        x2 = other.clone()
        x1.requires_grad = x2.requires_grad = True
        func(x1, x2).backward(target, retain_graph=True)
        return [x1.grad, x2.grad]

    grad_a = train(torch.nn.CosineSimilarity(dim=dim))
    grad_b = train(trident.CosineSimilarity(dim=dim))

    assert util.equal(grad_a[0], grad_b[0])
    assert util.equal(grad_a[1], grad_b[1])
