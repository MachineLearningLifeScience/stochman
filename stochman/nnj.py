#!/usr/bin/env python3
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple, Union

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
    """Abstract base class for activation functions.

    Any activation function subclassing this class will need to implement the
    `_jacobian` method for computing their Jacobian.
    """

    def __abstract_init__(self, activation, *args, **kwargs):
        activation.__init__(self, *args, **kwargs)
        self._activation = activation

    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = self._activation.forward(self, x)

        if jacobian:
            J, _ = self._jacobian(x, val)
            return val, J
        else:
            return val

    @abstractmethod
    def _jacobian(self, x: torch.Tensor, val: torch.Tensor):
        """Evaluate the Jacobian of an activation function.
        The Jacobian is evaluated at x, where the function
        attains value val."""
        pass

    def _jac_mul(
        self, x: torch.Tensor, val: torch.Tensor, Jseq: torch.Tensor, JseqType: JacType
    ) -> Tuple[torch.Tensor, JacType]:
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
            raise ValueError(
                "`ActivationJacobian:_jac_mul` method received an unknown "
                "jacobian type, should be either `JacType.Full` or `JacType.DIAG`"
            )
        return Jseq, jac_type


def _jac_mul_generic(J: torch.Tensor, Jseq: torch.Tensor, JseqType: JacType) -> Tuple[torch.Tensor, JacType]:
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
        raise ValueError(
            "`_jac_mul_generic` method received an unknown "
            "jacobian type, should be either `JacType.Full` or `JacType.DIAG`"
        )
    return Jseq, JacType.FULL


def _jac_add_generic(J1: torch.Tensor, J1Type: JacType, J2: torch.Tensor, J2Type: JacType):
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
        raise ValueError(
            "`_jac_add_generic` method received an unknown "
            "jacobian type, should be either `JacType.Full` or `JacType.DIAG`"
        )

    return J, JType


class Sequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, jacobian: bool = False, return_jac_type: bool = False):
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
            if JseqType is JacType.DIAG:
                Jseq = Jseq.view(xsh[:-1] + Jseq.shape[-1:])
            else:
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

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor):
        _, J, JType = self.forward(x, jacobian=True, return_jac_type=True)
        return J, JType

    def _jac_mul(self, x: torch.Tensor, val: torch.Tensor, Jseq: torch.Tensor, JseqType: JacType):
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
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)

    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = super().forward(x)

        if jacobian:
            J, _ = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor):
        b_sz = x.size()[0]
        J = self.weight.unsqueeze(0).repeat(b_sz, 1, 1)
        return J, JacType.FULL

    def _jac_mul(self, x: torch.Tensor, val: torch.Tensor, Jseq: torch.Tensor, JseqType: JacType):
        J, _ = self._jacobian(x, val)  # (batch)x(out)x(in)
        Jseq, JseqType = _jac_mul_generic(J, Jseq, JseqType)
        return Jseq, JseqType

    def inverse(self):
        inv = Linear(in_features=self.out_features, out_features=self.in_features, bias=self.bias is not None)
        pinv = self.weight.data.pinverse()
        inv.weight.data = nn.Parameter(pinv)
        if self.bias is not None:
            inv.bias = nn.Parameter(-pinv.mv(self.bias.data))
        return inv


class PosLinear(Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, jacobian: bool = False):
        if self.bias is None:
            val = F.linear(x, F.softplus(self.weight))
        else:
            val = F.linear(x, F.softplus(self.weight), F.softplus(self.bias))

        if jacobian:
            J, _ = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor):
        b_sz = x.shape[0]
        J = F.softplus(self.weight).unsqueeze(0).repeat(b_sz, 1, 1)
        return J, JacType.FULL

    def _jac_mul(self, x: torch.Tensor, val: torch.Tensor, Jseq: torch.Tensor, JseqType: JacType):
        J, _ = self._jacobian(x, val)  # (batch)x(out)x(in)
        Jseq, JseqType = _jac_mul_generic(J, Jseq, JseqType)
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

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor):
        J = 1.0 - val ** 2
        return J, JacType.DIAG

    def inverse(self):
        return ArcTanh()


class ArcTanh(ActivationJacobian, nn.Tanh):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor, jacobian: bool = False):
        xc = x.clamp(-(1 - 1e-4), 1 - 1e-4)  # the inverse is only defined on [-1, 1] so we project onto this interval
        val = (
            0.5 * (1.0 + xc).log() - 0.5 * (1.0 - xc).log()
        )  # XXX: is it stable to compute log((1+xc)/(1-xc)) ? (that would be faster)

        if jacobian:
            J, _ = self._jacobian(xc, val)
            return val, J
        else:
            return val

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor):
        J = -1.0 / (x ** 2 - 1.0)
        return J, JacType.DIAG


class Reciprocal(nn.Module, ActivationJacobian):
    def __init__(self, b: float = 0.0):
        super().__init__()
        self.b = b

    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = 1.0 / (x + self.b)

        if jacobian:
            J, _ = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor):
        J = -((val) ** 2)
        return J, JacType.DIAG


class OneMinusX(nn.Module, ActivationJacobian):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = 1 - x

        if jacobian:
            J, _ = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor):
        J = -torch.ones_like(x)
        return J, JacType.DIAG

    def _jac_mul(self, x: torch.Tensor, val: torch.Tensor, Jseq: torch.Tensor, JseqType: JacType):
        return -Jseq, JseqType


class Sqrt(nn.Module, ActivationJacobian):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = torch.sqrt(x)

        if jacobian:
            J, _ = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor):
        J = -0.5 / val
        return J, JacType.DIAG


class BatchNorm1d(ActivationJacobian, nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        ActivationJacobian.__abstract_init__(self, nn.BatchNorm1d, *args, **kwargs)

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor):
        B = x.shape[0]
        J = self.running_var.sqrt().repeat(B, 1)
        return J, JacType.DIAG


class ResidualBlock(nn.Module):
    def __init__(self, *args, in_features: Optional[int] = None, out_features: Optional[int] = None):
        super().__init__()

        # Are we given a sequence or should construct one?
        if len(args) == 1 and isinstance(args[0], (self.modules.container.Sequential, Sequential)):
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

    def forward(self, x: torch.Tensor, jacobian: bool = False):
        if self.apply_proj:
            val = self._projection(x) + self._F(x)
        else:
            val = x + self._F(x)

        if jacobian:
            J, _ = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor):
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

    def _jac_mul(self, x: torch.Tensor, val: torch.Tensor, Jseq, JseqType: JacType):
        JF, JFType = self._F._jac_mul(x, val, Jseq, JseqType)

        if self.apply_proj:
            JL, JLType = self._projection._jac_mul(x, val, Jseq, JseqType)
            J, JType = _jac_add_generic(JL, JLType, JF, JFType)
        else:
            J, JType = _jac_add_generic(Jseq, JseqType, JF, JFType)

        return J, JType


class Norm2(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = torch.sum(x ** 2, dim=self.dim, keepdim=True)

        if jacobian:
            J = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor):
        J = 2.0 * x.unsqueeze(1)
        return J, JacType.FULL

    def _jac_mul(self, x: torch.Tensor, val: torch.Tensor, Jseq, JseqType: JacType):
        J, _ = self._jacobian(x, val)  # (B)x(1)x(in) -- the current Jacobian
        return _jac_mul_generic(J, Jseq, JseqType)


class RBF(nn.Module):
    def __init__(
        self, dim: int, num_points: int, points: Optional[torch.Tensor] = None, beta: Union[torch.Tensor, float] = 1.0
    ):
        super().__init__()
        if points is None:
            self.points = nn.Parameter(torch.randn(num_points, dim))
        else:
            self.points = nn.Parameter(points, requires_grad=False)

        if isinstance(beta, torch.Tensor):
            self.beta = beta.view(1, -1)
        elif isinstance(beta, float):
            self.beta = beta
        else:
            raise ValueError(f"Expected parameter ``beta`` to either be a float or torch tensor but received {beta}")

    def __dist2__(self, x: torch.Tensor):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        points_norm = (self.points ** 2).sum(1).view(1, -1)
        d2 = x_norm + points_norm - 2.0 * torch.mm(x, self.points.transpose(0, 1))
        return d2.clamp(min=0.0)  # NxM

    def forward(self, x: torch.Tensor, jacobian: bool = False):
        D2 = self.__dist2__(x)  # (batch)-by-|x|-by-|points|
        val = torch.exp(-self.beta * D2)  # (batch)-by-|x|-by-|points|

        if jacobian:
            J = self._jacobian(x, val)
            return val, J
        else:
            return val

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor):
        T1 = -2.0 * self.beta * val  # BxNxM
        T2 = x.unsqueeze(1) - self.points.unsqueeze(0)
        J = T1.unsqueeze(-1) * T2
        return J, JacType.FULL

    def _jac_mul(self, x: torch.Tensor, val: torch.Tensor, Jseq, JseqType: JacType):
        J, _ = self._jacobian(x, val)  # (B)x(1)x(in) -- the current Jacobian
        return _jac_mul_generic(J, Jseq, JseqType)
