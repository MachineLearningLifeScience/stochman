#!/usr/bin/env python3
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from math import prod

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, jacobian=False):
        val = x
        
        if jacobian:
            xs = x.shape
            jac = torch.eye(prod(xs[1:]), prod(xs[1:])).repeat(xs[0], 1, 1).reshape(xs[0], *xs[1:], *xs[1:])
            return val, jac
        return val

    def _jacobian_mult(self, x, val, jac_in):
        return jac_in


def identity(x):
    m = Identity()
    return m(x, jacobian=True)[1]


class Sequential(nn.Sequential):
    def forward(self, x: Tensor, jacobian: Union[Tensor, bool] = False):
        if jacobian:
            j = identity(x) if (not isinstance(jacobian, Tensor) and jacobian) else jacobian
        for module in self._modules.values():
            val = module(x)
            if jacobian:
                j = module._jacobian_mult(x, val, j)
            x = val
        if jacobian:
            return x, j
        return x


class AbstractJacobian:
    def _jacobian(self, x, val):
        return self._jacobian_mult(x, val, identity(x))


class AbstractActivationJacobian:
    def _jacobian_mult(self, x, val, jac_in):
        jac = self._jacobian(x, val)
        n = jac_in.ndim - jac.ndim
        return jac_in * jac.reshape(jac.shape + (1,)*n)


class Linear(nn.Linear, AbstractJacobian):
    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = super().forward(x)

        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian_mult(self, x, val, jac_in):
        return F.linear(jac_in.movedim(1,-1), self.weight, bias=None).movedim(-1,1)


class Sigmoid(nn.Sigmoid, AbstractActivationJacobian):
    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = super().forward(x)

        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian(self, x, val):
        jac = val * (1.0 - val)
        return jac


class Upsample(nn.Upsample, AbstractJacobian):
    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = super().forward(x)

        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val  

    def _jacobian_mult(self, x, val, jac_in):
        xs = x.shape
        vs = val.shape
        if x.ndim == 3:
            return F.interpolate(jac_in.movedim((1,2),(-2,-1)).reshape(-1, *xs[1:]), 
                self.size, self.scale_factor, self.mode, self.align_corners
            ).reshape(xs[0], *jac_in.shape[3:], *vs[1:]).movedim((-2, -1), (1, 2))
        if x.ndim == 4:
            return F.interpolate(jac_in.movedim((1,2,3),(-3,-2,-1)).reshape(-1, *xs[1:]), 
                self.size, self.scale_factor, self.mode, self.align_corners
            ).reshape(xs[0], *jac_in.shape[4:], *vs[1:]).movedim((-3, -2, -1), (1, 2, 3))
        if x.ndim == 5:
            return F.interpolate(jac_in.movedim((1,2,3,4),(-4,-3,-2,-1)).reshape(-1, *xs[1:]), 
                self.size, self.scale_factor, self.mode, self.align_corners
            ).reshape(xs[0], *jac_in.shape[5:], *vs[1:]).movedim((-4,-3,-2, -1), (1, 2, 3, 4))


class Conv1d(nn.Conv1d, AbstractJacobian):
    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = super().forward(x)

        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian_mult(self, x, val, jac_in):
        b, c1, l1 = x.shape
        c2, l2 = val.shape[1:]
        return F.conv1d(jac_in.movedim((1, 2), (-2, -1)).reshape(-1, c1, l1), weight=self.weight, 
            bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups,
        ).reshape(b, *jac_in.shape[3:], c2, l2).movedim((-2, -1), (1, 2))


class ConvTranspose1d(nn.ConvTranspose1d, AbstractJacobian):
    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = super().forward(x)

        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian_mult(self, x, val, jac_in):
        b, c1, l1 = x.shape
        c2, l2 = val.shape[1:]
        return F.conv_transpose1d(jac_in.movedim((1, 2), (-2, -1)).reshape(-1, c1, l1), weight=self.weight, 
            bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups,
            output_padding=self.output_padding
        ).reshape(b, *jac_in.shape[3:], c2, l2).movedim((-2, -1), (1, 2))


class Conv2d(nn.Conv2d, AbstractJacobian):
    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = super().forward(x)

        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian_mult(self, x, val, jac_in):
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]
        return F.conv2d(jac_in.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c1, h1, w1), weight=self.weight, 
            bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups,
        ).reshape(b, *jac_in.shape[4:], c2, h2, w2).movedim((-3, -2, -1), (1, 2, 3))


class ConvTranspose2d(nn.ConvTranspose2d, AbstractJacobian):
    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = super().forward(x)

        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian_mult(self, x, val, jac_in):
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]
        return F.conv_transpose2d(jac_in.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c1, h1, w1), weight=self.weight, 
            bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups,
            output_padding=self.output_padding,
        ).reshape(b, *jac_in.shape[4:], c2, h2, w2).movedim((-3, -2, -1), (1, 2, 3))


class Conv3d(nn.Conv3d, AbstractJacobian):
    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = super().forward(x)

        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian_mult(self, x, val, jac_in):
        b, c1, d1, h1, w1 = x.shape
        c2, d2, h2, w2 = val.shape[1:]
        return F.conv3d(jac_in.movedim((1, 2, 3, 4), (-4, -3, -2, -1)).reshape(-1, c1, d1, h1, w1), weight=self.weight, 
            bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups,
        ).reshape(b, *jac_in.shape[5:], c2, d2, h2, w2).movedim((-4, -3, -2, -1), (1, 2, 3, 4))


class ConvTranspose3d(nn.ConvTranspose3d, AbstractJacobian):
    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = super().forward(x)

        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian_mult(self, x, val, jac_in):
        b, c1, d1, h1, w1 = x.shape
        c2, d2, h2, w2 = val.shape[1:]
        return F.conv_transpose3d(jac_in.movedim((1, 2, 3, 4), (-4, -3, -2, -1)).reshape(-1, c1, d1, h1, w1), weight=self.weight, 
            bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups,
            output_padding=self.output_padding
        ).reshape(b, *jac_in.shape[5:], c2, d2, h2, w2).movedim((-4, -3, -2, -1), (1, 2, 3, 4))


class Reshape(nn.Module, AbstractJacobian):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor, jacobian: bool = False) -> torch.Tensor:
        val = x.reshape(x.shape[0], *self.dims)
        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian_mult(self, x, val, jac_in):
        return jac_in.reshape(jac_in.shape[0], *self.dims, *jac_in.shape[2:])


class Flatten(nn.Module, AbstractJacobian):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, jacobian: bool = False) -> torch.Tensor:
        val = x.reshape(x.shape[0], -1)
        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian_mult(self, x, val, jac_in):
        if jac_in.ndim == 5:  # 1d conv
            return jac_in.reshape(jac_in.shape[0], -1, *jac_in.shape[3:])
        if jac_in.ndim == 7:  # 2d conv
            return jac_in.reshape(jac_in.shape[0], -1, *jac_in.shape[4:])
        if jac_in.ndim == 9:  # 3d conv
            return jac_in.reshape(jac_in.shape[0], -1, *jac_in.shape[5:])


class PosLinear(Linear):
    def forward(self, x: torch.Tensor, jacobian: bool = False):
        if self.bias is None:
            val = F.linear(x, F.softplus(self.weight))
        else:
            val = F.linear(x, F.softplus(self.weight), F.softplus(self.bias))

        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian_mult(self, x, val, jac_in):
        return F.linear(jac_in.movedim(1,-1), F.softplus(self.weight), bias=None).movedim(-1,1)


class ReLU(nn.ReLU, AbstractActivationJacobian):
    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = super().forward(x)

        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian(self, x, val):
        jac = (val > 0.0).type(val.dtype)
        return jac


class ELU(nn.ELU, AbstractActivationJacobian):
    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = super().forward(x)

        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian(self, x, val):
        jac = torch.ones_like(val)
        jac[x <= 0.0] = val[x <= 0.0] + self.alpha
        return jac


class Hardshrink(nn.Hardshrink, AbstractActivationJacobian):
    def forward(self, x: torch.Tensor, jacobian: bool = False):
        val = super().forward(x)

        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val

    def _jacobian(self, x, val):
        jac = torch.ones_like(val)
        jac[-self.lambd < x < self.lambd] = 0.0
        return jac


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
