#!/usr/bin/env python3
from abc import ABC, abstractmethod
from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn


class JacType(Enum):
    """Class for declaring the type of an intermediate Jacobian.
    Options are:
        DIAG:   The Jacobian is a (B)x(N) matrix that represents
                a diagonal matrix of size (B)x(N)x(N)
        FULL:   The Jacobian is a matrix of whatever size.
    """

    DIAG = 1
    FULL = 2


class ActivationJacobian(ABC):
    """Abstract class for activation functions.

    Any activation functions subclassing this class will need to implement
    a method for computing their Jacobian."""

    def __abstract_init__(self, activation, *args, **kwargs):
        activation.__init__(self, *args, **kwargs)
        self.__activation__ = activation

    def forward(self, x, jacobian=False):
        val = self.__activation__.forward(self, x)

        if jacobian:
            J, _ = self._jacobian(x, val)
            return val, J
        else:
            return val

    @abstractmethod
    def _jacobian(self, x, val):
        """Evaluate the Jacobian of an activation function.
        The Jacobian is evaluated at x, where the function
        attains value val."""
        pass

    def _jac_mul(self, x, val, Jseq, JseqType):
        """Multiply the Jacobian at x with M.
        This can potentially be done more efficiently than
        first computing the Jacobian, and then performing the
        multiplication."""

        J, _ = self._jacobian(x, val)  # (B)x(in) -- the current Jacobian;
        # should be interpreted as a diagonal matrix (B)x(in)x(in)

        # We either want to (matrix) multiply the current diagonal Jacobian with a
        #  *) vector: size (B)x(in)
        #       This should be interpreted as a diagonal matrix of size
        #       (B)x(in)x(in), and we want to perform the product diag(J) * diag(Jseq)
        #  *) matrix: size (B)x(in)x(M)
        #       In this case we want to return diag(J) * Jseq

        if JseqType is JacType.FULL:  # Jseq.dim() is 3: # Jseq is matrix
            Jseq = torch.einsum("bi,bim->bim", J, Jseq)  # diag(J) * Jseq
            # (B)x(in) * (B)x(in)x(M)-> (B)x(in)x(M)
            jac_type = JacType.FULL
        elif JseqType is JacType.DIAG:  # Jseq.dim() is 2: # Jseq is vector (representing a diagonal matrix)
            Jseq = J * Jseq  # diag(J) * diag(Jseq)
            # (B)x(in) * (B)x(in) -> (B)x(in)
            jac_type = JacType.DIAG
        else:
            print("ActivationJacobian:_jac_mul: What the hell?")
        return Jseq, jac_type


def __jac_mul_generic__(J, Jseq, JseqType):
    #
    # J: (B)x(K)x(in) -- the current Jacobian

    # We either want to (matrix) multiply the current Jacobian with a
    #  *) vector: size (B)x(in)
    #       This should be interpreted as a diagonal matrix of size
    #       (B)x(in)x(in), and we want to perform the product diag(J) * diag(Jseq)
    #  *) matrix: size (B)x(in)x(M)
    #       In this case we want to return diag(J) * Jseq
    if JseqType is JacType.FULL:  # Jseq.dim() is 3: # Jseq is matrix
        Jseq = torch.einsum("bki,bim->bkm", J, Jseq)  # J * Jseq
        # (B)x(K)(in) * (B)x(in)x(M)-> (B)x(K)x(M)
    elif JseqType is JacType.DIAG:  # Jseq.dim() is 2: # Jseq is vector (representing a diagonal matrix)
        Jseq = torch.einsum("bki,bi->bki", J, Jseq)  # J * diag(Jseq)
        # (B)x(K)(in) * (B)x(in) -> (B)x(K)x(in)
    else:
        print("__jac_mul_generic__: What the hell?")
    return Jseq, JacType.FULL


def __jac_add_generic__(J1, J1Type, J2, J2Type):
    # Add two Jacobians of possibly different types
    if J1Type is J2Type:
        J = J1 + J2
        JType = J1Type
    elif J1Type is JacType.FULL and J2Type is JacType.DIAG:
        J = J1 + torch.diag_embed(J2)
        JType = JacType.FULL
    elif J1Type is JacType.DIAG and J2Type is JacType.FULL:
        J = torch.diag_embed(J1) + J2
        JType = JacType.FULL
    else:
        print("__jac_add_generic__: What the ....?")

    return J, JType


class Sequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, jacobian=False, return_jac_type=False):
        xsh = x.shape
        if len(xsh) == 1:
            x = x.unsqueeze(0)

        x = x.view(-1, xsh[-1])  # Nx(d)

        if jacobian:
            Jseq = None

            for module in self._modules.values():
                val = module(x)
                if Jseq is None:
                    Jseq, JseqType = module._jacobian(x, val)
                else:
                    Jseq, JseqType = module._jac_mul(x, val, Jseq, JseqType)
                x = val

            x = x.view(xsh[:-1] + torch.Size([x.shape[-1]]))
            Jseq = Jseq.view(xsh[:-1] + Jseq.shape[-2:])

            if return_jac_type:
                return x, Jseq, JseqType
            else:
                return x, Jseq
        else:
            for module in self._modules.values():
                x = module(x)
            x = x.view(xsh[:-1] + torch.Size([x.shape[-1]]))

            return x

    def _jacobian(self, x, val):
        _, J, JType = self.forward(x, jacobian=True, return_jac_type=True)
        return J, JType

    def _jac_mul(self, x, val, Jseq, JseqType):
        for module in self._modules.values():
            val = module(x)
            Jseq, JseqType = module._jac_mul(x, val, Jseq, JseqType)
            x = val
        return Jseq, JseqType

    def inverse(self):
        layers = [L.inverse() for L in reversed(self._modules.values())]
        return Sequential(*layers)

    def dimensions(self):
        in_features, out_features = None, None
        for module in self._modules.values():
            if in_features is None and hasattr(module, "__constants__") and "in_features" in module.__constants__:
                in_features = module.in_features
            if hasattr(module, "__constants__") and "out_features" in module.__constants__:
                out_features = module.out_features
        return in_features, out_features

    def disable_training(self):
        state = []
        for module in self._modules.values():
            state.append(module.training)
            module.training = False
        return state

    def enable_training(self, state=None):
        if state is None:
            for module in self._modules.values():
                module.training = True
        else:
            for module, new_state in zip(self._modules.values(), state):
                module.training = new_state


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def forward(self, x, jacobian=False):
        val = super().forward(x)

        if jacobian:
            J, _ = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(self, x, val):
        b_sz = x.size()[0]
        J = self.weight.unsqueeze(0).repeat(b_sz, 1, 1)
        return J, JacType.FULL

    def _jac_mul(self, x, val, Jseq, JseqType):
        J, _ = self._jacobian(x, val)  # (batch)x(out)x(in)
        Jseq, JseqType = __jac_mul_generic__(J, Jseq, JseqType)
        return Jseq, JseqType

    def inverse(self):
        I = Linear(in_features=self.out_features, out_features=self.in_features, bias=self.bias is not None)
        pinv = self.weight.data.pinverse()
        I.weight.data = nn.Parameter(pinv)
        if self.bias is not None:
            I.bias = nn.Parameter(-pinv.mv(self.bias.data))
        return I


class PosLinear(Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, jacobian=False):
        if self.bias is None:
            val = F.linear(x, F.softplus(self.weight))
        else:
            val = F.linear(x, F.softplus(self.weight), F.softplus(self.bias))

        if jacobian:
            J, _ = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(self, x, val):
        b_sz = x.shape[0]
        J = F.softplus(self.weight).unsqueeze(0).repeat(b_sz, 1, 1)
        return J, JacType.FULL

    def _jac_mul(self, x, val, Jseq, JseqType):
        J, _ = self._jacobian(x, val)  # (batch)x(out)x(in)
        Jseq, JseqType = __jac_mul_generic__(J, Jseq, JseqType)
        return Jseq, JseqType


class ReLU(ActivationJacobian, nn.ReLU):
    def __init__(self, *args, **kwargs):
        ActivationJacobian.__abstract_init__(self, nn.ReLU, *args, **kwargs)

    def _jacobian(self, x, val):
        J = (val > 0.0).type(val.dtype)  # XXX: can we avoid the type cast (it's expensive) ?
        return J, JacType.DIAG


class ELU(ActivationJacobian, nn.ELU):
    def __init__(self, *args, **kwargs):
        ActivationJacobian.__abstract_init__(self, nn.ELU, *args, **kwargs)

    def _jacobian(self, x, val):
        J = torch.ones_like(val)
        J[val < 0.0] = val[val < 0.0] + self.alpha
        return J, JacType.DIAG


class Hardshrink(ActivationJacobian, nn.Hardshrink):
    def __init__(self, *args, **kwargs):
        ActivationJacobian.__abstract_init__(self, nn.Hardshrink, *args, **kwargs)

    def _jacobian(self, x, val):
        J = torch.ones_like(val)
        # J[(-self.lambd < x) & (x < self.lambd)] = 0.0
        J[val.abs() < 1e-3] = 0.0
        return J, JacType.DIAG


class Hardtanh(ActivationJacobian, nn.Hardtanh):
    def __init__(self, *args, **kwargs):
        ActivationJacobian.__abstract_init__(self, nn.Hardtanh, *args, **kwargs)

    def _jacobian(self, x, val):
        J = torch.zeros_like(val)
        # J[(self.min_val < x) & (x < self.max_val)] = 1.0
        J[val.abs() < 1.0] = 1.0
        return J, JacType.DIAG


class LeakyReLU(ActivationJacobian, nn.LeakyReLU):
    def __init__(self, *args, **kwargs):
        ActivationJacobian.__abstract_init__(self, nn.LeakyReLU, *args, **kwargs)

    def _jacobian(self, x, val):
        J = torch.ones_like(val)
        J[val < 0.0] = self.negative_slope
        return J, JacType.DIAG


class Sigmoid(ActivationJacobian, nn.Sigmoid):
    def __init__(self, *args, **kwargs):
        ActivationJacobian.__abstract_init__(self, nn.Sigmoid, *args, **kwargs)

    def _jacobian(self, x, val):
        J = val * (1.0 - val)
        return J, JacType.DIAG


class Softplus(ActivationJacobian, nn.Softplus):
    def __init__(self, *args, **kwargs):
        ActivationJacobian.__abstract_init__(self, nn.Softplus, *args, **kwargs)

    def _jacobian(self, x, val):
        J = torch.sigmoid(self.beta * x)
        return J, JacType.DIAG


class Tanh(ActivationJacobian, nn.Tanh):
    def __init__(self, *args, **kwargs):
        ActivationJacobian.__abstract_init__(self, nn.Tanh, *args, **kwargs)

    def _jacobian(self, x, val):
        J = 1.0 - val ** 2
        return J, JacType.DIAG

    def inverse(self):
        return ArcTanh()


class ArcTanh(ActivationJacobian, nn.Tanh):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, jacobian=False):
        xc = x.clamp(-(1 - 1e-4), 1 - 1e-4)  # the inverse is only defined on [-1, 1] so we project onto this interval
        val = (
            0.5 * (1.0 + xc).log() - 0.5 * (1.0 - xc).log()
        )  # XXX: is it stable to compute log((1+xc)/(1-xc)) ? (that would be faster)

        if jacobian:
            J, _ = self._jacobian(xc, val)
            return val, J
        else:
            return val

    def _jacobian(self, xc, val):
        J = -1.0 / (xc ** 2 - 1.0)
        return J, JacType.DIAG


class Reciprocal(nn.Module, ActivationJacobian):
    def __init__(self, b=0.0):
        super().__init__()
        self.b = b

    def forward(self, x, jacobian=False):
        val = 1.0 / (x + self.b)

        if jacobian:
            J, _ = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(self, x, val):
        J = -((val) ** 2)
        return J, JacType.DIAG


class OneMinusX(nn.Module, ActivationJacobian):
    def __init__(self):
        super().__init__()

    def forward(self, x, jacobian=False):
        val = 1 - x

        if jacobian:
            J, _ = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(self, x, val):
        J = -torch.ones_like(x)
        return J, JacType.DIAG

    def _jac_mul(self, x, val, Jseq, JseqType):
        return -Jseq, JseqType


class Sqrt(nn.Module, ActivationJacobian):
    def __init__(self):
        super().__init__()

    def forward(self, x, jacobian=False):
        val = torch.sqrt(x)

        if jacobian:
            J, _ = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(self, x, val):
        J = -0.5 / val
        return J, JacType.DIAG


class BatchNorm1d(ActivationJacobian, nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        ActivationJacobian.__abstract_init__(self, nn.BatchNorm1d, *args, **kwargs)

    def _jacobian(self, x, val):
        B = x.shape[0]
        J = self.running_var.sqrt().repeat(B, 1)
        return J, JacType.DIAG


class ResidualBlock(nn.Module):
    def __init__(self, *args, in_features=None, out_features=None):
        super().__init__()

        # Are we given a sequence or should construct one?
        if len(args) is 1 and isinstance(args[0], (modules.container.Sequential, Sequential)):
            self._F = args[0]
        else:
            self._F = Sequential(*args)

        # Are input/output dimensions given? (If so, we will perform projection)
        est_in_features, est_out_features = self._F.dimensions()
        if in_features is None and out_features is None and est_in_features != est_out_features:
            in_features, out_features = est_in_features, est_out_features
        self.apply_proj = in_features is not None and out_features is not None
        if self.apply_proj:
            self._projection = Linear(in_features, out_features)

    def forward(self, x, jacobian=False):
        if self.apply_proj:
            val = self._projection(x) + self._F(x)
        else:
            val = x + self._F(x)

        if jacobian:
            J, _ = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(self, x, val):
        J, JType = self._F._jacobian(x, val)

        if self.apply_proj:
            if JType is JacType.DIAG:
                J = torch.diag_embed(J)
                JType = JacType.FULL
            J += self._projection.weight
        else:
            if JType is JacType.DIAG:
                J += 1.0
            else:  # JacType.FULL
                J += torch.eye(J.shape[0])

        return J, JType

    def _jac_mul(self, x, val, Jseq, JseqType):
        JF, JFType = self._F._jac_mul(x, val, Jseq, JseqType)

        if self.apply_proj:
            JL, JLType = self._projection._jac_mul(x, val, Jseq, JseqType)
            J, JType = __jac_add_generic__(JL, JLType, JF, JFType)
        else:
            J, JType = __jac_add_generic__(Jseq, JseqType, JF, JFType)

        return J, JType


class Norm2(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x, jacobian=False):
        val = torch.sum(x ** 2, dim=self.dim, keepdim=True)

        if jacobian:
            J = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(self, x, val):
        J = 2.0 * x.unsqueeze(1)
        return J, JacType.FULL

    def _jac_mul(self, x, val, Jseq, JseqType):
        J, _ = self._jacobian(x, val)  # (B)x(1)x(in) -- the current Jacobian
        return __jac_mul_generic__(J, Jseq, JseqType)


class RBF(nn.Module):
    def __init__(self, dim, num_points, points=None, beta=1.0):
        super().__init__()
        if points is None:
            self.points = nn.Parameter(torch.randn(num_points, dim))
        else:
            self.points = nn.Parameter(points, requires_grad=False)
        if isinstance(beta, torch.Tensor):
            self.beta = beta.view(1, -1)
        else:
            self.beta = beta

    def __dist2__(self, x):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        points_norm = (self.points ** 2).sum(1).view(1, -1)
        d2 = x_norm + points_norm - 2.0 * torch.mm(x, self.points.transpose(0, 1))
        return d2.clamp(min=0.0)  # NxM
        # if x.dim() is 2:
        #    x = x.unsqueeze(0) # BxNxD
        # x_norm = (x**2).sum(-1, keepdim=True) # BxNx1
        # points_norm = (self.points**2).sum(-1, keepdim=True).view(1, 1, -1) # 1x1xM
        # d2 = x_norm + points_norm - 2.0 * torch.bmm(x, self.points.t().unsqueeze(0).expand(x.shape[0], -1, -1))
        # return d2.clamp(min=0.0) # BxNxM

    def forward(self, x, jacobian=False):
        D2 = self.__dist2__(x)  # (batch)-by-|x|-by-|points|
        val = torch.exp(-self.beta * D2)  # (batch)-by-|x|-by-|points|

        if jacobian:
            J = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(self, x, val):
        T1 = -2.0 * self.beta * val  # BxNxM
        T2 = x.unsqueeze(1) - self.points.unsqueeze(0)
        J = T1.unsqueeze(-1) * T2
        return J, JacType.FULL

    def _jac_mul(self, x, val, Jseq, JseqType):
        J, _ = self._jacobian(x, val)  # (B)x(1)x(in) -- the current Jacobian
        return __jac_mul_generic__(J, Jseq, JseqType)


def fd_jacobian(function, x, h=1e-4):
    """Compute finite difference Jacobian of given function
    at a single location x. This function is mainly considered
    useful for debugging."""

    no_batch = x.dim() is 1
    if no_batch:
        x = x.unsqueeze(0)
    elif x.dim() > 2:
        raise Exception("The input should be a D-vector or a BxD matrix")
    B, D = x.shape

    # Compute finite differences
    E = h * torch.eye(D)
    try:
        # Disable "training" in the function (relevant eg. for batch normalization)
        orig_state = function.disable_training()
        Jnum = torch.cat([((function(x[b] + E) - function(x[b].unsqueeze(0))).t() / h).unsqueeze(0) for b in range(B)])
    finally:
        function.enable_training(orig_state)  # re-enable training

    if no_batch:
        Jnum = Jnum.squeeze(0)

    return Jnum


def jacobian_check(function, in_dim=None, h=1e-4, verbose=True):
    """Accepts an nnj module and checks the
    Jacobian via the finite differences method.

    Args:
        function:   An nnj module object. The
                    function to be tested.

    Returns a tuple of the following form:
    (Jacobian_analytical, Jacobian_finite_differences)
    """

    with torch.no_grad():
        batch_size = 5
        if in_dim is None:
            in_dim, _ = functions.dimensions()
            if in_dim is None:
                in_dim = 10
        x = torch.randn(batch_size, in_dim)
        try:
            orig_state = function.disable_training()
            y, J, Jtype = function(x, jacobian=True, return_jac_type=True)
        finally:
            function.enable_training(orig_state)

        if Jtype is JacType.DIAG:
            J = J.diag_embed()

        Jnum = fd_jacobian(function, x)

        if verbose:
            residual = (J - Jnum).abs().max()
            if residual > 100 * h:
                print("****** Warning: exceedingly large error:", residual.item(), "******")
            else:
                print("OK (residual = ", residual.item(), ")")
        else:
            return J, Jnum


def test(models=None):
    in_features = 10
    if models is None:
        models = [
            Sequential(Linear(in_features, 2), Softplus(beta=100, threshold=5), Linear(2, 4), Tanh()),
            Sequential(RBF(in_features, 30), Linear(30, 2)),
            Sequential(Linear(in_features, 4), Norm2()),
            Sequential(Linear(in_features, 50), ReLU(), Linear(50, 100), Softplus()),
            Sequential(Linear(in_features, 256)),
            Sequential(Softplus(), Linear(in_features, 3), Softplus()),
            Sequential(Softplus(), Sigmoid(), Linear(in_features, 3)),
            Sequential(Softplus(), Sigmoid()),
            Sequential(Linear(in_features, 3), OneMinusX()),
            Sequential(PosLinear(in_features, 2), Softplus(beta=100, threshold=5), PosLinear(2, 4), Tanh()),
            Sequential(PosLinear(in_features, 5), Reciprocal(b=1.0)),
            Sequential(ReLU(), ELU(), LeakyReLU(), Sigmoid(), Softplus(), Tanh()),
            Sequential(ReLU()),
            Sequential(ELU()),
            Sequential(LeakyReLU()),
            Sequential(Sigmoid()),
            Sequential(Softplus()),
            Sequential(Tanh()),
            Sequential(Hardshrink()),
            Sequential(Hardtanh()),
            Sequential(ResidualBlock(Linear(in_features, 50), ReLU())),
            Sequential(BatchNorm1d(in_features)),
            Sequential(
                BatchNorm1d(in_features),
                ResidualBlock(Linear(in_features, 25), Softplus()),
                BatchNorm1d(25),
                ResidualBlock(Linear(25, 25), Softplus()),
            ),
        ]
    for model in models:
        jacobian_check(model, in_features)


if __name__ == "__main__":
    # test()
    in_dim = 10
    latent_dim = 2
    C = torch.randn(3, 2)
    mdl = Sequential(
        RBF(latent_dim, num_points=3, points=C, beta=0.9),
        PosLinear(C.shape[0], 1, bias=False),
        Reciprocal(b=1e-4),
        PosLinear(1, in_dim),
    )

    x = torch.randn(50, 2)

    y, J = mdl(x, True)
    print(y.shape)
    print(J.shape)
