import numpy
import pytest
import torch

from stochman import nnj

_batch_size = 2
_features = 5
_dims = 10

_linear_input = torch.randn(_batch_size, _features)
_1d_conv_input = torch.randn(_batch_size, _features, _dims)
_2d_conv_input = torch.randn(_batch_size, _features, _dims, _dims)
_3d_conv_input = torch.randn(_batch_size, _features, _dims, _dims, _dims)


def _compare_jacobian(f, x):
    out = f(x)
    output = torch.autograd.functional.jacobian(f, x)   
    m = out.ndim
    output = output.movedim(m,1)
    res = torch.stack([output[i,i] for i in range(_batch_size)], dim=0)
    return res


"""
_models = [
    nnj.Sequential(
        nnj.Linear(_in_features, 2), nnj.Softplus(beta=100, threshold=5), nnj.Linear(2, 4), nnj.Tanh()
    ),
    nnj.Sequential(nnj.RBF(_in_features, 30), nnj.Linear(30, 2)),
    nnj.Sequential(nnj.Linear(_in_features, 4), nnj.Norm2()),
    nnj.Sequential(nnj.Linear(_in_features, 50), nnj.ReLU(), nnj.Linear(50, 100), nnj.Softplus()),
    nnj.Sequential(nnj.Linear(_in_features, 256)),
    nnj.Sequential(nnj.Softplus(), nnj.Linear(_in_features, 3), nnj.Softplus()),
    nnj.Sequential(nnj.Softplus(), nnj.Sigmoid(), nnj.Linear(_in_features, 3)),
    nnj.Sequential(nnj.Softplus(), nnj.Sigmoid()),
    nnj.Sequential(nnj.Linear(_in_features, 3), nnj.OneMinusX()),
    nnj.Sequential(
        nnj.PosLinear(_in_features, 2), nnj.Softplus(beta=100, threshold=5), nnj.PosLinear(2, 4), nnj.Tanh()
    ),
    nnj.Sequential(nnj.PosLinear(_in_features, 5), nnj.Reciprocal(b=1.0)),
    nnj.Sequential(nnj.ReLU(), nnj.ELU(), nnj.LeakyReLU(), nnj.Sigmoid(), nnj.Softplus(), nnj.Tanh()),
    nnj.Sequential(nnj.ReLU()),
    nnj.Sequential(nnj.ELU()),
    nnj.Sequential(nnj.LeakyReLU()),
    nnj.Sequential(nnj.Sigmoid()),
    nnj.Sequential(nnj.Softplus()),
    nnj.Sequential(nnj.Tanh()),
    nnj.Sequential(nnj.Hardshrink()),
    nnj.Sequential(nnj.Hardtanh()),
    nnj.Sequential(nnj.ResidualBlock(nnj.Linear(_in_features, 50), nnj.ReLU())),
    nnj.Sequential(nnj.BatchNorm1d(_in_features)),
    nnj.Sequential(
        nnj.BatchNorm1d(_in_features),
        nnj.ResidualBlock(nnj.Linear(_in_features, 25), nnj.Softplus()),
        nnj.BatchNorm1d(25),
        nnj.ResidualBlock(nnj.Linear(25, 25), nnj.Softplus()),
    ),
]
"""



@pytest.mark.parametrize("model, input", 
    [
        (nnj.Sequential(nnj.Linear(_features, 2)), _linear_input),
        (nnj.Sequential(nnj.Linear(_features, 2), nnj.Sigmoid()), _linear_input),
        (nnj.Sequential(nnj.Linear(_features, 5), nnj.Sigmoid(), nnj.Linear(5, 2)), _linear_input),
        (nnj.Sequential(
            nnj.Linear(_features, 2), nnj.Softplus(beta=100, threshold=5), nnj.Linear(2, 4), nnj.Tanh()
        ), _linear_input),
        (nnj.Sequential(nnj.Linear(_features, 2), nnj.Sigmoid(), nnj.ReLU()), _linear_input),
        (nnj.Sequential(nnj.Conv1d(_features, 2, 5)), _1d_conv_input),
        (nnj.Sequential(nnj.Conv2d(_features, 2, 5)), _2d_conv_input),
        (nnj.Sequential(nnj.Conv3d(_features, 2, 5)), _3d_conv_input),
        (nnj.Sequential(
            nnj.Linear(_features, 8), nnj.Sigmoid(), nnj.Reshape(2, 4), nnj.Conv1d(2, 1, 2),
        ),_linear_input),
        (nnj.Sequential(
            nnj.Linear(_features, 32), nnj.Sigmoid(), nnj.Reshape(2, 4, 4), nnj.Conv2d(2, 1, 2),
        ),_linear_input),
        (nnj.Sequential(
            nnj.Linear(_features, 128), nnj.Sigmoid(), nnj.Reshape(2, 4, 4, 4), nnj.Conv3d(2, 1, 2),
        ),_linear_input),
        (nnj.Sequential(
            nnj.Conv1d(_features, 2, 3), nnj.Flatten(), nnj.Linear(8*2, 5), nnj.ReLU(),
        ),_1d_conv_input),
        (nnj.Sequential(
            nnj.Conv2d(_features, 2, 3), nnj.Flatten(), nnj.Linear(8*8*2, 5), nnj.ReLU(),
        ),_2d_conv_input),
        (nnj.Sequential(
            nnj.Conv3d(_features, 2, 3), nnj.Flatten(), nnj.Linear(8*8*8*2, 5), nnj.ReLU(),
        ),_3d_conv_input)
    ]
)
class TestJacobian:
    def test_jacobians(self, model, input):
        """Test that the analytical jacobian of the model is consistent with finite
        order approximation
        """
        _, jac = model(input, jacobian=True)
        jacnum = _compare_jacobian(model, input)
        assert torch.isclose(jac, jacnum, atol=1e-7).all(), "jacobians did not match"

    @pytest.mark.parametrize("return_jac", [True, False])
    def test_jac_return(self, model, input, return_jac):
        """ Test that all models returns the jacobian output if asked for it """
        output = model(input, jacobian=return_jac)
        if return_jac:
            assert len(output) == 2, "expected two outputs when jacobian=True"
            assert all(isinstance(o, torch.Tensor) for o in output), "expected all outputs to be torch tensors"
        else:
            assert isinstance(output, torch.Tensor)

