import pytest
import torch

import trident
from tests import util


@pytest.mark.parametrize(
    "x_shape, dim",
    [((1431, 500, 200), 2), ((21, 6400), 1), ((86,), 0)],
)
def test_forward(x_shape, dim, device):
    factory_kwargs = {"device": device}
    x1 = torch.randn(x_shape, **factory_kwargs)
    x2 = torch.randn(x_shape, **factory_kwargs)

    assert util.equal(
        torch.nn.functional.cosine_similarity(x1, x2, dim),
        trident.function.cosine_similarity(x1, x2, dim),
    )


@pytest.mark.parametrize(
    "x_shape, dim",
    [((1280, 1000, 200), 1), ((640, 21), 0), ((90,), 0)],
)
def test_backward(x_shape, dim, device):
    factory_kwargs = {"device": device}
    x1 = torch.randn(x_shape, **factory_kwargs)
    x2 = torch.randn(x_shape, **factory_kwargs)

    def get_output_shape(x_shape, dim):
        if len(x_shape) == 1:
            return ()
        elif len(x_shape) == 2:
            return x_shape[1] if dim == 0 else x_shape[0]
        else:
            if dim == 0:
                return (x_shape[1], x_shape[2])
            elif dim == 1:
                return (x_shape[0], x_shape[2])
            else:
                return (x_shape[0], x_shape[1])

    grad_output = torch.randn(get_output_shape(x_shape, dim), **factory_kwargs)

    def train(func):
        i = x1.clone()
        j = x2.clone()
        i.requires_grad = j.requires_grad = True
        func(i, j).backward(grad_output, retain_graph=True)

        return i.grad, j.grad

    (x, y) = train(torch.nn.CosineSimilarity(dim))
    (a, b) = train(trident.CosineSimilarity(dim))

    assert util.equal(x, a)
    assert util.equal(y, b)


@pytest.mark.parametrize("z_size, y_size, x_size, dim", [(640, 21, 86, 2)])
def test_cosine_similarity(z_size, y_size, x_size, dim, device, dtype):
    factory_kwargs = {"device": device, "dtype": dtype}
    x1 = torch.randn(z_size, y_size, x_size, **factory_kwargs, requires_grad=True)
    x2 = torch.randn(z_size, y_size, x_size, **factory_kwargs, requires_grad=True)

    output = trident.CosineSimilarity(dim).forward(x1, x2)
    assert output is not None
    assert output.dtype == dtype

    if dim == 0:
        output_shape = (y_size, x_size)
    elif dim == 1:
        output_shape = (z_size, x_size)
    else:
        output_shape = (z_size, y_size)

    grad_output = torch.randn(output_shape, **factory_kwargs)

    output.backward(grad_output)
    assert x1.grad is not None
    assert x1.grad.dtype == dtype
    assert x2.grad is not None
    assert x2.grad.dtype == dtype
