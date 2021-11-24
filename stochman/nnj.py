from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from math import prod


class Identity(nn.Module):
    """ Identity module that will return the same input as it receives. """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, jacobian: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        val = x

        if jacobian:
            xs = x.shape
            jac = torch.eye(prod(xs[1:]), prod(xs[1:]), dtype=x.dtype).repeat(xs[0], 1, 1).reshape(xs[0], *xs[1:], *xs[1:])
            return val, jac
        return val

    def _jacobian_mult(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        return jac_in


def identity(x: Tensor) -> Tensor:
    """ Function that for a given input x returns the corresponding identity jacobian matrix """
    m = Identity()
    return m(x, jacobian=True)[1]


class Sequential(nn.Sequential):
    """ Subclass of sequential that also supports calculating the jacobian through an network """

    def forward(
        self, x: Tensor, jacobian: Union[Tensor, bool] = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
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
    """Abstract class that will overwrite the default behaviour of the forward method such that it
    is also possible to return the jacobian
    """

    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        return self._jacobian_mult(x, val, identity(x))

    def __call__(self, x: Tensor, jacobian: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        val = self._call_impl(x)
        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val


class Linear(AbstractJacobian, nn.Linear):
    def _jacobian_mult(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        return F.linear(jac_in.movedim(1, -1), self.weight, bias=None).movedim(-1, 1)


class PosLinear(AbstractJacobian, nn.Linear):
    def forward(self, x: Tensor):
        bias = F.softplus(self.bias) if self.bias is not None else self.bias
        val = F.linear(x, F.softplus(self.weight), bias)
        return val

    def _jacobian_mult(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        return F.linear(jac_in.movedim(1, -1), F.softplus(self.weight), bias=None).movedim(-1, 1)


class Upsample(AbstractJacobian, nn.Upsample):
    def _jacobian_mult(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        xs = x.shape
        vs = val.shape

        dims1 = tuple(range(1, x.ndim))
        dims2 = tuple(range(-x.ndim + 1, 0))

        return (
            F.interpolate(
                jac_in.movedim(dims1, dims2).reshape(-1, *xs[1:]),
                self.size,
                self.scale_factor,
                self.mode,
                self.align_corners,
            )
            .reshape(xs[0], *jac_in.shape[x.ndim :], *vs[1:])
            .movedim(dims2, dims1)
        )


class Conv1d(AbstractJacobian, nn.Conv1d):
    def _jacobian_mult(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, l1 = x.shape
        c2, l2 = val.shape[1:]
        return (
            F.conv1d(
                jac_in.movedim((1, 2), (-2, -1)).reshape(-1, c1, l1),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            .reshape(b, *jac_in.shape[3:], c2, l2)
            .movedim((-2, -1), (1, 2))
        )


class ConvTranspose1d(AbstractJacobian, nn.ConvTranspose1d):
    def _jacobian_mult(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, l1 = x.shape
        c2, l2 = val.shape[1:]
        return (
            F.conv_transpose1d(
                jac_in.movedim((1, 2), (-2, -1)).reshape(-1, c1, l1),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                output_padding=self.output_padding,
            )
            .reshape(b, *jac_in.shape[3:], c2, l2)
            .movedim((-2, -1), (1, 2))
        )


class Conv2d(AbstractJacobian, nn.Conv2d):
    def _jacobian_mult(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]
        return (
            F.conv2d(
                jac_in.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c1, h1, w1),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            .reshape(b, *jac_in.shape[4:], c2, h2, w2)
            .movedim((-3, -2, -1), (1, 2, 3))
        )


class ConvTranspose2d(AbstractJacobian, nn.ConvTranspose2d):
    def _jacobian_mult(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]
        return (
            F.conv_transpose2d(
                jac_in.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c1, h1, w1),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                output_padding=self.output_padding,
            )
            .reshape(b, *jac_in.shape[4:], c2, h2, w2)
            .movedim((-3, -2, -1), (1, 2, 3))
        )


class Conv3d(AbstractJacobian, nn.Conv3d):
    def _jacobian_mult(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, d1, h1, w1 = x.shape
        c2, d2, h2, w2 = val.shape[1:]
        return (
            F.conv3d(
                jac_in.movedim((1, 2, 3, 4), (-4, -3, -2, -1)).reshape(-1, c1, d1, h1, w1),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            .reshape(b, *jac_in.shape[5:], c2, d2, h2, w2)
            .movedim((-4, -3, -2, -1), (1, 2, 3, 4))
        )


class ConvTranspose3d(AbstractJacobian, nn.ConvTranspose3d):
    def _jacobian_mult(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, d1, h1, w1 = x.shape
        c2, d2, h2, w2 = val.shape[1:]
        return (
            F.conv_transpose3d(
                jac_in.movedim((1, 2, 3, 4), (-4, -3, -2, -1)).reshape(-1, c1, d1, h1, w1),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                output_padding=self.output_padding,
            )
            .reshape(b, *jac_in.shape[5:], c2, d2, h2, w2)
            .movedim((-4, -3, -2, -1), (1, 2, 3, 4))
        )


class Reshape(AbstractJacobian, nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        val = x.reshape(x.shape[0], *self.dims)
        return val

    def _jacobian_mult(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        return jac_in.reshape(jac_in.shape[0], *self.dims, *jac_in.shape[2:])


class Flatten(AbstractJacobian, nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        val = x.reshape(x.shape[0], -1)
        return val

    def _jacobian_mult(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        if jac_in.ndim == 5:  # 1d conv
            return jac_in.reshape(jac_in.shape[0], -1, *jac_in.shape[3:])
        if jac_in.ndim == 7:  # 2d conv
            return jac_in.reshape(jac_in.shape[0], -1, *jac_in.shape[4:])
        if jac_in.ndim == 9:  # 3d conv
            return jac_in.reshape(jac_in.shape[0], -1, *jac_in.shape[5:])


class AbstractActivationJacobian:
    def _jacobian_mult(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        jac = self._jacobian(x, val)
        n = jac_in.ndim - jac.ndim
        return jac_in * jac.reshape(jac.shape + (1,) * n)

    def __call__(self, x: Tensor, jacobian: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        val = self._call_impl(x)
        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val


class Sigmoid(AbstractActivationJacobian, nn.Sigmoid):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = val * (1.0 - val)
        return jac


class ReLU(AbstractActivationJacobian, nn.ReLU):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = (val > 0.0).type(val.dtype)
        return jac


class ELU(AbstractActivationJacobian, nn.ELU):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = torch.ones_like(val)
        jac[x <= 0.0] = val[x <= 0.0] + self.alpha
        return jac


class Hardshrink(AbstractActivationJacobian, nn.Hardshrink):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = torch.ones_like(val)
        jac[torch.logical_and(-self.lambd < x, x < self.lambd)] = 0.0
        return jac


class Hardtanh(AbstractActivationJacobian, nn.Hardtanh):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = torch.zeros_like(val)
        jac[val.abs() < 1.0] = 1.0
        return jac


class LeakyReLU(AbstractActivationJacobian, nn.LeakyReLU):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = torch.zeros_like(val)
        jac[val.abs() < 1.0] = 1.0
        return jac


class Softplus(AbstractActivationJacobian, nn.Softplus):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = torch.sigmoid(self.beta * x)
        return jac


class Tanh(AbstractActivationJacobian, nn.Tanh):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = 1.0 - val ** 2
        return jac


class ArcTanh(AbstractActivationJacobian, nn.Tanh):
    def forward(self, x: Tensor) -> Tensor:
        xc = x.clamp(
            -(1 - 1e-4), 1 - 1e-4
        )  # the inverse is only defined on [-1, 1] so we project onto this interval
        val = (
            0.5 * (1.0 + xc).log() - 0.5 * (1.0 - xc).log()
        )  # XXX: is it stable to compute log((1+xc)/(1-xc)) ? (that would be faster)
        return val

    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = -1.0 / (x ** 2 - 1.0)
        return jac


class Reciprocal(AbstractActivationJacobian, nn.Module):
    def __init__(self, b: float = 0.0):
        super().__init__()
        self.b = b

    def forward(self, x: Tensor) -> Tensor:
        val = 1.0 / (x + self.b)
        return val

    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = -((val) ** 2)
        return jac


class OneMinusX(AbstractActivationJacobian, nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        val = 1 - x
        return val

    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = -torch.ones_like(x)
        return jac


class Sqrt(AbstractActivationJacobian, nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        val = torch.sqrt(x)
        return val

    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = -0.5 / val
        return jac
