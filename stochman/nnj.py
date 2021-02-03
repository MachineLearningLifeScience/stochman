#!/usr/bin/env python3
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class JacType(Enum):
    """Class for declaring the type of an intermediate Jacobian.
    Options are:
        DIAG:   The Jacobian is a (B)x(N) matrix that represents
                a diagonal matrix of size (B)x(N)x(N)
        FULL:   The Jacobian is a matrix of whatever size.
    """

    DIAG = 'diag'
    FULL = 'full'
    CONV = 'conv'
    
    def __eq__(self, other: Union[str, Enum]) -> bool:
        other = other.value if isinstance(other, Enum) else str(other)
        return self.value.lower() == other.lower()
    

class Jacobian(torch.Tensor):
    """ Class representing a jacobian tensor, subclasses from torch.Tensor 
        Requires the additional `jactype` parameter to initialize, which
        is a string indicating the jacobian type
    """
    def __init__(self, tensor, jactype):
        available_jactype = [item.value for item in JacType]
        if jactype not in available_jactype:
            raise ValueError(f'Tried to initialize jacobian tensor with unknown jacobian type {jactype}.'
                             f' Please choose between {available_jactype}')
        self.jactype = jactype
    
    @staticmethod
    def __new__(cls, x, jactype, *args, **kwargs):
        cls.jactype = jactype
        return super().__new__(cls, x, *args, **kwargs)
    
    def __repr__(self):
        tensor_repr = super().__repr__()
        tensor_repr = tensor_repr.replace('tensor', 'jacobian')
        tensor_repr += f'\n jactype={self.jactype.value if isinstance(self.jactype, Enum) else self.jactype}'
        return tensor_repr
    
    def __add__(self, other):
        if isinstance(other, Jacobian):            
            if self.jactype == other.jactype:
                res = torch.add(self, other)
                return jacobian(res, self.jactype)
            if self.jactype == JacType.FULL and other.jactype == JacType.DIAG:
                res = torch.add(self, torch.diag_embed(other))
                return jacobian(res, JacType.FULL)
            if self.jactype == JacType.DIAG and other.jactype == JacType.FULL:
                res = torch.add(torch.diag_embed(self), other)
                return jacobian(res, JacType.FULL)                
            if self.jactype == JacType.CONV and other.jactype == JacType.CONV:
                res = torch.add(self, other)
                return jacobian(res, JacType.CONV)
            raise ValueError('Unknown addition of jacobian matrices')
        
        return super().__add__(other)
        
    def __matmul__(self, other):
        if isinstance(other, Jacobian):
            # diag * diag
            if self.jactype == JacType.DIAG and other.jactype == JacType.DIAG:
                res = self * other
                return jacobian(res, JacType.DIAG)
            # full * full
            if self.jactype == JacType.FULL and other.jactype == JacType.FULL:
                res = torch.matmul(self, other)
                return jacobian(res, JacType.FULL)
            # diag * full
            if self.jactype == JacType.DIAG and other.jactype == JacType.FULL:
                res = torch.matmul(torch.diag_embed(self), other)
                return jacobian(res, JacType.FULL)
            # full * diag
            if self.jactype == JacType.FULL and other.jactype == JacType.DIAG:
                res = torch.matmul(self, torch.diag_embed(other))
                return jacobian(res, JacType.FULL)
            if self.jactype == JacType.CONV:
                # conv * conv
                if other == JacType.CONV:
                    res = self * other
                    return jacobian(res, JacType.CONV)
           
        raise ValueError('Unknown matrix multiplication of jacobian matrices')
    

def jacobian(tensor, jactype):
    """ Initialize a jacobian tensor by a specified jacobian type """
    return Jacobian(tensor, jactype)


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
    def _jacobian(self, x: torch.Tensor, val: torch.Tensor) -> Jacobian:
        """Evaluate the Jacobian of an activation function.
        The Jacobian is evaluated at x, where the function
        attains value val."""
        pass

    def _jac_mul(
        self, x: torch.Tensor, val: torch.Tensor, jac_in: torch.Tensor) -> Jacobian:
        """Multiply the Jacobian at x with M.
        This can potentially be done more efficiently than
        first computing the Jacobian, and then performing the
        multiplication."""
        jac = self._jacobian(x, val)  # (B)x(in) -- the current Jacobian;
        return jac @ jac_in


class Sequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, jacobian: bool = False):
        xsh = x.shape
        if len(xsh) == 1:
            x = x.unsqueeze(0)

        x = x.view(-1, xsh[-1])  # Nx(d)

        if jacobian:
            jac = None

            for module in self._modules.values():
                val = module(x)
                if jac is None:
                    jac = module._jacobian(x, val)
                else:
                    jac = module._jac_mul(x, val, jac)
                x = val

            x = x.view(xsh[:-1] + torch.Size([x.shape[-1]]))
            if jac.jactype == JacType.DIAG:
                jac = jac.view(xsh[:-1] + jac.shape[-1:])
            else:
                jac = jac.view(xsh[:-1] + jac.shape[-2:])

            return x, jac

        for module in self._modules.values():
            x = module(x)
        x = x.view(xsh[:-1] + torch.Size([x.shape[-1]]))
        return x

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor) -> Jacobian:
        _, J = self.forward(x, jacobian=True)
        return J

    def _jac_mul(self, x: torch.Tensor, val: torch.Tensor, jac: torch.Tensor) -> Jacobian:
        for module in self._modules.values():
            val = module(x)
            jac = module._jac_mul(x, val, jac)
            x = val
        return jac

    def inverse(self):
        layers = [L.inverse() for L in reversed(self._modules.values())]
        return Sequential(*layers)

    def dimensions(self):
        in_features, out_features = None, None
        for module in self._modules.values():
            if (
                in_features is None
                and hasattr(module, "__constants__")
                and "in_features" in module.__constants__
            ):
                in_features = module.in_features
            if hasattr(module, "__constants__") and "out_features" in module.__constants__:
                out_features = module.out_features
        return in_features, out_features


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)

    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = super().forward(x)

        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor) -> Jacobian:
        b_sz = x.size()[0]
        jac = self.weight.unsqueeze(0).repeat(b_sz, 1, 1)
        return jacobian(jac, JacType.FULL)

    def _jac_mul(self, x: torch.Tensor, val: torch.Tensor, jac_in: torch.Tensor) -> Jacobian:
        jac = self._jacobian(x, val)  # (batch)x(out)x(in)
        return jac @ jac_in

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
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor) -> Jacobian:
        b_sz = x.shape[0]
        jac = F.softplus(self.weight).unsqueeze(0).repeat(b_sz, 1, 1)
        return jacobian(jac, JacType.FULL)

    def _jac_mul(self, x: torch.Tensor, val: torch.Tensor, jac_in: torch.Tensor) -> Jacobian:
        jac = self._jacobian(x, val)  # (batch)x(out)x(in)
        return jac @ jac_in


class ReLU(ActivationJacobian, nn.ReLU):
    def __init__(self, *args, **kwargs):
        ActivationJacobian.__abstract_init__(self, nn.ReLU, *args, **kwargs)

    def _jacobian(self, x, val) -> Jacobian:
        jac = (val > 0.0).type(val.dtype)
        return jacobian(jac, JacType.DIAG)


class ELU(ActivationJacobian, nn.ELU):
    def __init__(self, *args, **kwargs):
        ActivationJacobian.__abstract_init__(self, nn.ELU, *args, **kwargs)

    def _jacobian(self, x, val) -> Jacobian:
        jac = torch.ones_like(val)
        jac[val < 0.0] = val[val < 0.0] + self.alpha
        return jacobian(jac, JacType.DIAG)


class Hardshrink(ActivationJacobian, nn.Hardshrink):
    def __init__(self, *args, **kwargs):
        ActivationJacobian.__abstract_init__(self, nn.Hardshrink, *args, **kwargs)

    def _jacobian(self, x, val) -> Jacobian:
        jac = torch.ones_like(val)
        # J[(-self.lambd < x) & (x < self.lambd)] = 0.0
        jac[val.abs() < 1e-3] = 0.0
        return jacobian(jac, JacType.DIAG)


class Hardtanh(ActivationJacobian, nn.Hardtanh):
    def __init__(self, *args, **kwargs):
        ActivationJacobian.__abstract_init__(self, nn.Hardtanh, *args, **kwargs)

    def _jacobian(self, x, val) -> Jacobian:
        jac = torch.zeros_like(val)
        # J[(self.min_val < x) & (x < self.max_val)] = 1.0
        jac[val.abs() < 1.0] = 1.0
        return jacobian(jac, JacType.DIAG)


class LeakyReLU(ActivationJacobian, nn.LeakyReLU):
    def __init__(self, *args, **kwargs):
        ActivationJacobian.__abstract_init__(self, nn.LeakyReLU, *args, **kwargs)

    def _jacobian(self, x, val) -> Jacobian:
        jac = torch.ones_like(val)
        jac[val < 0.0] = self.negative_slope
        return jacobian(jac, JacType.DIAG)


class Sigmoid(ActivationJacobian, nn.Sigmoid):
    def __init__(self, *args, **kwargs):
        ActivationJacobian.__abstract_init__(self, nn.Sigmoid, *args, **kwargs)

    def _jacobian(self, x, val) -> Jacobian:
        jac = val * (1.0 - val)
        return jacobian(jac, JacType.DIAG)


class Softplus(ActivationJacobian, nn.Softplus):
    def __init__(self, *args, **kwargs):
        ActivationJacobian.__abstract_init__(self, nn.Softplus, *args, **kwargs)

    def _jacobian(self, x, val) -> Jacobian:
        jac = torch.sigmoid(self.beta * x)
        return jacobian(jac, JacType.DIAG)


class Tanh(ActivationJacobian, nn.Tanh):
    def __init__(self, *args, **kwargs):
        ActivationJacobian.__abstract_init__(self, nn.Tanh, *args, **kwargs)

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor) -> Jacobian:
        jac = 1.0 - val ** 2
        return jacobian(jac, JacType.DIAG)

    def inverse(self):
        return ArcTanh()


class ArcTanh(ActivationJacobian, nn.Tanh):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor, jacobian: bool = False):
        xc = x.clamp(
            -(1 - 1e-4), 1 - 1e-4
        )  # the inverse is only defined on [-1, 1] so we project onto this interval
        val = (
            0.5 * (1.0 + xc).log() - 0.5 * (1.0 - xc).log()
        )  # XXX: is it stable to compute log((1+xc)/(1-xc)) ? (that would be faster)

        if jacobian:
            jac = self._jacobian(xc, val)
            return val, jac
        return val

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor) -> Jacobian:
        jac = -1.0 / (x ** 2 - 1.0)
        return jacobian(jac, JacType.DIAG)


class Reciprocal(nn.Module, ActivationJacobian):
    def __init__(self, b: float = 0.0):
        super().__init__()
        self.b = b

    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = 1.0 / (x + self.b)

        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor) -> Jacobian:
        jac = -((val) ** 2)
        return jacobian(jac, JacType.DIAG)


class OneMinusX(nn.Module, ActivationJacobian):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = 1 - x

        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor) -> Jacobian:
        jac = -torch.ones_like(x)
        return jacobian(jac, JacType.DIAG)

    def _jac_mul(self, x: torch.Tensor, val: torch.Tensor, jac_in: torch.Tensor) -> Jacobian:
        return -jac_in


class Sqrt(nn.Module, ActivationJacobian):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = torch.sqrt(x)

        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor) -> Jacobian:
        jac = -0.5 / val
        return jacobian(jac, JacType.DIAG)


class BatchNorm1d(ActivationJacobian, nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        ActivationJacobian.__abstract_init__(self, nn.BatchNorm1d, *args, **kwargs)

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor) -> Jacobian:
        B = x.shape[0]
        jac = self.running_var.sqrt().repeat(B, 1)
        return jacobian(jac, JacType.DIAG)


class ResidualBlock(nn.Module):
    def __init__(self, *args, in_features: Optional[int] = None, out_features: Optional[int] = None):
        super().__init__()

        # Are we given a sequence or should construct one?
        if len(args) == 1 and isinstance(args[0], Sequential):
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
            J = self._jacobian(x, val)
            return val, J
        return val

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor) -> Jacobian:
        jac = self._F._jacobian(x, val)

        if self.apply_proj:
            if jac.jactype == JacType.DIAG:
                jac = torch.diag_embed(jac)
            jac += self._projection.weight
        else:
            if jac.jactype is JacType.DIAG:
                jac += 1.0
            else:  # JacType.FULL
                jac += torch.eye(jac.shape[0])

        return jac

    def _jac_mul(self, x: torch.Tensor, val: torch.Tensor, jac_in: torch.Tensor) -> Jacobian:
        jac = self._F._jac_mul(x, val, jac_in)

        if self.apply_proj:
            jac_in = self._projection._jac_mul(x, val, jac_in)
        return jac + jac_in


class Norm2(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = torch.sum(x ** 2, dim=self.dim, keepdim=True)

        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor):
        jac = 2.0 * x.unsqueeze(1)
        return jacobian(jac, JacType.FULL)

    def _jac_mul(self, x: torch.Tensor, val: torch.Tensor, jac_in: torch.Tensor) -> Jacobian:
        jac = self._jacobian(x, val)  # (B)x(1)x(in) -- the current Jacobian
        return jac @ jac_in


class RBF(nn.Module):
    def __init__(
        self,
        dim: int,
        num_points: int,
        points: Optional[torch.Tensor] = None,
        beta: Union[torch.Tensor, float] = 1.0,
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
            raise ValueError(
                f"Expected parameter ``beta`` to either be a float or torch tensor but received {beta}"
            )

    def __dist2__(self, x: torch.Tensor) -> torch.Tensor:
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
        return val

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor) -> Jacobian:
        t1 = -2.0 * self.beta * val  # BxNxM
        t2 = x.unsqueeze(1) - self.points.unsqueeze(0)
        jac = t1.unsqueeze(-1) * t2
        return jacobian(jac, JacType.FULL)

    def _jac_mul(self, x: torch.Tensor, val: torch.Tensor, jac_in: torch.Tensor) -> Jacobian:
        jac = self._jacobian(x, val)  # (B)x(1)x(in) -- the current Jacobian
        return jac @ jac_in


class _BaseJacConv:
    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = super().forward(x)
        if jacobian:
            jac = self._jacobian(x, val)
            val = (val, jac)
        return val

    def _jacobian(self, x: torch.Tensor, val: torch.Tensor) -> Jacobian:
        w = self._conv_to_toeplitz(x.shape[1:])
        w = w.unsqueeze(0).repeat(x.shape[0], 1, 1)
        return jacobian(w, JacType.CONV)        

    
class Conv1d(_BaseJacConv, nn.Conv1d):
    def _conv_to_toeplitz(self, input_shape):
        identity = torch.eye(np.prod(input_shape).item()).reshape([-1] + list(input_shape))
        output = F.conv1d(identity, self.weight, None, self.stride, self.padding)
        W = output.reshape(output.shape[0], -1).T
        return W


class Conv2d(_BaseJacConv, nn.Conv2d):
    def _conv_to_toeplitz(self, input_shape):
        identity = torch.eye(np.prod(input_shape).item()).reshape([-1] + list(input_shape))
        output = F.conv2d(identity, self.weight, None, self.stride, self.padding)
        W = output.reshape(output.shape[0], -1).T
        return W


class Conv2D(_BaseJacConv, nn.Conv3d):
    def _conv_to_toeplitz(self, input_shape):
        identity = torch.eye(np.prod(input_shape).item()).reshape([-1] + list(input_shape))
        output = F.conv3d(identity, self.weight, None, self.stride, self.padding)
        W = output.reshape(output.shape[0], -1).T
        return W
