import pytest
import torch

import trident
from tests import util


@pytest.mark.parametrize(
    "z_size, y_size, x_size, dim",
    [(1431, 500, 200, 0), (221, 1250, 200, 1), (21, 6400, 86, 2)],
)
def test_forward(z_size, y_size, x_size, dim, device):
    factory_kwargs = {"device": device}

    x1 = torch.randn(z_size, y_size, x_size, **factory_kwargs)
    x2 = torch.randn(z_size, y_size, x_size, **factory_kwargs)

    assert util.equal(
        torch.nn.functional.cosine_similarity(x1, x2, dim=dim),
        trident.function.cosine_similarity(x1, x2, dim=dim),
    )


@pytest.mark.parametrize(
    "z_size, y_size, x_size, dim",
    [(1280, 1000, 200, 0), (200, 1280, 200, 1), (640, 21, 86, 2)],
)
def test_backward(z_size, y_size, x_size, dim, device):
    factory_kwargs = {"device": device}

    x1 = torch.randn(z_size, y_size, x_size, **factory_kwargs)
    x2 = torch.randn(z_size, y_size, x_size, **factory_kwargs)

    if dim == 0:
        target_dim = (y_size, x_size)
    elif dim == 1:
        target_dim = (z_size, x_size)
    else:
        target_dim = (z_size, y_size)

    grad_ouput = torch.randn(target_dim, **factory_kwargs)

    def train(func):
        i = x1.clone()
        j = x2.clone()
        i.requires_grad = j.requires_grad = True
        func(i, j).backward(grad_ouput, retain_graph=True)
        return i.grad, j.grad

    (x, y) = train(torch.nn.CosineSimilarity(dim))
    (a, b) = train(trident.CosineSimilarity(dim))

    assert util.equal(x, a)
    assert util.equal(y, b)
