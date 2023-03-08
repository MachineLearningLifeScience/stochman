from copy import deepcopy
from typing import Callable

import pytest
import torch

from stochman import nnj

_ = torch.manual_seed(42)

_batch_size = 2
_features = 5
_dims = 6

_linear_input_shape = (_batch_size, _features)
_1d_conv_input_shape = (_batch_size, _features, _dims)
_2d_conv_input_shape = (_batch_size, _features, _dims, _dims)
_3d_conv_input_shape = (_batch_size, _features, _dims, _dims, _dims)


def _compare_jacobian(f: Callable, x: torch.Tensor) -> torch.Tensor:
    """Use pytorch build-in Jacobian function to compare for correctness of computations"""
    out = f(x)
    output = torch.autograd.functional.jacobian(f, x)
    m = out.ndim
    output = output.movedim(m, 1)
    res = torch.stack([output[i, i] for i in range(_batch_size)], dim=0)
    return res


@pytest.mark.parametrize(
    "model, input_shape",
    [
        (nnj.Sequential(nnj.Identity(), nnj.Identity()), _linear_input_shape),
        (nnj.Linear(_features, 2), _linear_input_shape),
        (nnj.Sequential(nnj.PosLinear(_features, 2), nnj.Reciprocal()), _linear_input_shape),
        (nnj.Sequential(nnj.Linear(_features, 2), nnj.Sigmoid(), nnj.ArcTanh()), _linear_input_shape),
        (nnj.Sequential(nnj.Linear(_features, 5), nnj.Sigmoid(), nnj.Linear(5, 2)), _linear_input_shape),
        (
            nnj.Sequential(nnj.Linear(_features, 2), nnj.Softplus(beta=100, threshold=5), nnj.Linear(2, 4)),
            _linear_input_shape,
        ),
        (
            nnj.Sequential(
                nnj.ELU(),
                nnj.Linear(_features, 2),
                nnj.Sigmoid(),
                nnj.ReLU(),
                nnj.Sqrt(),
                nnj.Hardshrink(),
            ),
            _linear_input_shape,
        ),
        (nnj.Sequential(nnj.Linear(_features, 2), nnj.LeakyReLU()), _linear_input_shape),
        (nnj.Sequential(nnj.Linear(_features, 2), nnj.Tanh()), _linear_input_shape),
        (nnj.Sequential(nnj.Linear(_features, 2), nnj.OneMinusX()), _linear_input_shape),
        (nnj.Sequential(nnj.Linear(_features, 2), nnj.PReLU()), _linear_input_shape),
        (nnj.Sequential(nnj.Linear(_features, 2), nnj.Softmax(dim=-1)), _linear_input_shape),
        (nnj.Sequential(nnj.Linear(_features, 2), nnj.PReLU()), _linear_input_shape),
        (
            nnj.Sequential(nnj.Conv1d(_features, 2, 5), nnj.ConvTranspose1d(2, _features, 5)),
            _1d_conv_input_shape,
        ),
        (
            nnj.Sequential(nnj.Conv2d(_features, 2, 5), nnj.ConvTranspose2d(2, _features, 5)),
            _2d_conv_input_shape,
        ),
        (
            nnj.Sequential(nnj.Conv3d(_features, 2, 5), nnj.ConvTranspose3d(2, _features, 5)),
            _3d_conv_input_shape,
        ),
        (
            nnj.Sequential(
                nnj.Linear(_features, 8),
                nnj.Sigmoid(),
                nnj.Reshape(2, 4),
                nnj.Conv1d(2, 1, 2),
            ),
            _linear_input_shape,
        ),
        (
            nnj.Sequential(
                nnj.Linear(_features, 32),
                nnj.Sigmoid(),
                nnj.Reshape(2, 4, 4),
                nnj.Conv2d(2, 1, 2),
            ),
            _linear_input_shape,
        ),
        (
            nnj.Sequential(
                nnj.Linear(_features, 128),
                nnj.Sigmoid(),
                nnj.Reshape(2, 4, 4, 4),
                nnj.Conv3d(2, 1, 2),
            ),
            _linear_input_shape,
        ),
        (
            nnj.Sequential(
                nnj.Conv1d(_features, 2, 3),
                nnj.Flatten(),
                nnj.Linear(4 * 2, 5),
                nnj.Softmax(dim=-1),
            ),
            _1d_conv_input_shape,
        ),
        (
            nnj.Sequential(
                nnj.Conv2d(_features, 2, 3),
                nnj.Flatten(),
                nnj.Linear(4 * 4 * 2, 5),
                nnj.Softmax(dim=-1),
            ),
            _2d_conv_input_shape,
        ),
        (
            nnj.Sequential(
                nnj.Conv3d(_features, 2, 3),
                nnj.Flatten(),
                nnj.Linear(4 * 4 * 4 * 2, 5),
                nnj.Softmax(dim=-1),
            ),
            _3d_conv_input_shape,
        ),
        (
            nnj.Sequential(nnj.Conv2d(_features, 2, 3), nnj.Hardtanh(), nnj.Upsample(scale_factor=2)),
            _2d_conv_input_shape,
        ),
        (nnj.Sequential(nnj.Conv1d(_features, 3, 3), nnj.BatchNorm1d(3)), _1d_conv_input_shape),
        (nnj.Sequential(nnj.Conv2d(_features, 3, 3), nnj.BatchNorm2d(3)), _2d_conv_input_shape),
        (nnj.Sequential(nnj.Conv3d(_features, 3, 3), nnj.BatchNorm3d(3)), _3d_conv_input_shape),
        (nnj.Sequential(nnj.Conv1d(_features, 3, 3), nnj.MaxPool1d(2)), _1d_conv_input_shape),
        (nnj.Sequential(nnj.Conv2d(_features, 3, 3), nnj.MaxPool2d(2)), _2d_conv_input_shape),
        (nnj.Sequential(nnj.Conv3d(_features, 3, 3), nnj.MaxPool3d(2)), _3d_conv_input_shape),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
class TestJacobian:
    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    def test_jacobians(self, model, input_shape, device, dtype):
        """Test that the analytical Jacobian of the model is consistent with finite
        order approximation
        """
        if "cuda" in device and not torch.cuda.is_available():
            pytest.skip("Test requires cuda support")

        model = deepcopy(model).to(device=device, dtype=dtype).eval()
        input = torch.randn(*input_shape, device=device, dtype=dtype)
        _, jac = model(input, jacobian=True)
        jacnum = _compare_jacobian(model, input).to(device)
        assert torch.isclose(jac, jacnum, atol=1e-3).all(), "Jacobians did not match"

    @pytest.mark.parametrize("return_jac", [True, False])
    def test_jac_return(self, model, input_shape, device, return_jac):
        """Test that all models returns the jacobian output if asked for it"""
        if "cuda" in device and not torch.cuda.is_available():
            pytest.skip("Test requires cuda support")

        input = torch.randn(*input_shape, device=device)
        model = deepcopy(model).to(device)
        output = model(input, jacobian=return_jac)
        if return_jac:
            assert len(output) == 2, "expected two outputs when jacobian=True"
            assert all(
                isinstance(o, torch.Tensor) for o in output
            ), "expected all outputs to be torch tensors"
            assert all(str(o.device) == device for o in output)
        else:
            assert isinstance(output, torch.Tensor)
            assert str(output.device) == device


def test_jac_works_in_separate_sequentials():
    """Tests if you can provide a Jacobian tensor as a kwarg"""
    first_sequential = nnj.Sequential(nnj.Linear(3, 2), nnj.Tanh())
    second_sequential = nnj.Sequential(nnj.Linear(2, 1), nnj.Tanh())

    inputs = torch.randn((10, 3))
    hidden_values, first_jacobian = first_sequential(inputs, jacobian=True)
    _, _ = second_sequential(hidden_values, jacobian=first_jacobian)
