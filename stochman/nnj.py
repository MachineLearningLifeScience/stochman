from builtins import breakpoint
from math import prod
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Identity(nn.Module):
    """Identity module that will return the same input as it receives."""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, jacobian: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        val = x

        if jacobian:
            xs = x.shape
            jac = (
                torch.eye(prod(xs[1:]), prod(xs[1:]), dtype=x.dtype, device=x.device)
                .repeat(xs[0], 1, 1)
                .reshape(xs[0], *xs[1:], *xs[1:])
            )
            return val, jac
        return val

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        return jac_in


def identity(x: Tensor) -> Tensor:
    """Function that for a given input x returns the corresponding identity jacobian matrix"""
    m = Identity()
    return m(x, jacobian=True)[1]


class Sequential(nn.Sequential):
    """Subclass of sequential that also supports calculating the jacobian through an network"""

    def forward(
        self, x: Tensor, jacobian: Union[Tensor, bool] = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if jacobian:
            j = identity(x) if (not isinstance(jacobian, Tensor) and jacobian) else jacobian
        for module in self._modules.values():
            val = module(x)
            if jacobian:
                j = module._jacobian_wrt_input_mult_left_vec(x, val, j)
            x = val
        if jacobian:
            return x, j
        return x


class AbstractJacobian:
    """Abstract class that will overwrite the default behaviour of the forward method such that it
    is also possible to return the jacobian
    """

    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        return self._jacobian_wrt_input_mult_left_vec(x, val, identity(x))

    def __call__(self, x: Tensor, jacobian: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        val = self._call_impl(x)
        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val


class Linear(AbstractJacobian, nn.Linear):
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        return F.linear(jac_in.movedim(1, -1), self.weight, bias=None).movedim(-1, 1)

    def _jacobian_wrt_input_transpose_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        return F.linear(jac_in.movedim(1, -1), self.weight.T, bias=None).movedim(-1, 1)

    def _jacobian_wrt_input(self, x: Tensor, val: Tensor) -> Tensor:
        return self.weight

    def _jacobian_wrt_weight(self, x: Tensor, val: Tensor) -> Tensor:
        b, c1 = x.shape
        c2 = val.shape[1]
        out_identity = torch.diag_embed(torch.ones(c2, device=x.device))
        jacobian = torch.einsum("bk,ij->bijk", x, out_identity).reshape(b, c2, c2 * c1)
        if self.bias is not None:
            jacobian = torch.cat([jacobian, out_identity.unsqueeze(0).expand(b, -1, -1)], dim=2)
        return jacobian

    def _jacobian_wrt_input_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_full(x, val, tmp)
        elif not diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_diag(x, val, tmp)
        elif diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_full(x, val, tmp)
        elif diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, tmp)

    def _jacobian_wrt_weight_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return self._jacobian_wrt_weight_sandwich_full_to_full(x, val, tmp)
        elif not diag_inp and diag_out:
            return self._jacobian_wrt_weight_sandwich_full_to_diag(x, val, tmp)
        elif diag_inp and not diag_out:
            return self._jacobian_wrt_weight_sandwich_diag_to_full(x, val, tmp)
        elif diag_inp and diag_out:
            return self._jacobian_wrt_weight_sandwich_diag_to_diag(x, val, tmp)

    def _jacobian_wrt_input_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        return torch.einsum("nm,bnj,jk->bmk", self.weight, tmp, self.weight)

    def _jacobian_wrt_input_sandwich_full_to_diag(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        return torch.einsum("nm,bnj,jm->bm", self.weight, tmp, self.weight)

    def _jacobian_wrt_input_sandwich_diag_to_full(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        return torch.einsum("nm,bn,nk->bmk", self.weight, tmp_diag, self.weight)

    def _jacobian_wrt_input_sandwich_diag_to_diag(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        return torch.einsum("nm,bn,nm->bm", self.weight, tmp_diag, self.weight)

    def _jacobian_wrt_weight_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        jacobian = self._jacobian_wrt_weight(x, val)
        return torch.einsum("bji,bjk,bkq->biq", jacobian, tmp, jacobian)

    def _jacobian_wrt_weight_sandwich_full_to_diag(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        tmp_diag = torch.diagonal(tmp, dim1=1, dim2=2)
        return self._jacobian_wrt_weight_sandwich_diag_to_diag(x, val, tmp_diag)

    def _jacobian_wrt_weight_sandwich_diag_to_full(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        jacobian = self._jacobian_wrt_weight(x, val)
        return torch.einsum("bji,bj,bjq->biq", jacobian, tmp_diag, jacobian)

    def _jacobian_wrt_weight_sandwich_diag_to_diag(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:

        b, c1 = x.shape
        c2 = val.shape[1]

        Jt_tmp_J = torch.bmm(tmp_diag.unsqueeze(2), (x**2).unsqueeze(1)).view(b, c1 * c2)

        if self.bias is not None:
            Jt_tmp_J = torch.cat([Jt_tmp_J, tmp_diag], dim=1)

        return Jt_tmp_J


class PosLinear(AbstractJacobian, nn.Linear):
    def forward(self, x: Tensor):
        bias = F.softplus(self.bias) if self.bias is not None else self.bias
        val = F.linear(x, F.softplus(self.weight), bias)
        return val

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        return F.linear(jac_in.movedim(1, -1), F.softplus(self.weight), bias=None).movedim(-1, 1)


class Upsample(AbstractJacobian, nn.Upsample):
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
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

    def _jacobian_wrt_weight_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        # non parametric, so return empty
        return None

    def _jacobian_wrt_input_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_full(x, val, tmp)
        elif not diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_diag(x, val, tmp)
        elif diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_full(x, val, tmp)
        elif diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, tmp)

    def _jacobian_wrt_input_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        assert c1 == c2

        weight = torch.ones(1, 1, int(self.scale_factor), int(self.scale_factor), device=x.device)

        tmp = tmp.reshape(b, c2, h2 * w2, c2, h2 * w2)
        tmp = tmp.movedim(2, 3)
        tmp_J = F.conv2d(
            tmp.reshape(b * c2 * c2 * h2 * w2, 1, h2, w2),
            weight=weight,
            bias=None,
            stride=int(self.scale_factor),
            padding=0,
            dilation=1,
            groups=1,
        ).reshape(b * c2 * c2, h2 * w2, h1 * w1)

        Jt_tmpt = tmp_J.movedim(-1, -2)

        Jt_tmpt_J = F.conv2d(
            Jt_tmpt.reshape(b * c2 * c2 * h1 * w1, 1, h2, w2),
            weight=weight,
            bias=None,
            stride=int(self.scale_factor),
            padding=0,
            dilation=1,
            groups=1,
        ).reshape(b * c2 * c2, h1 * w1, h1 * w1)

        Jt_tmp_J = Jt_tmpt_J.movedim(-1, -2)

        Jt_tmp_J = Jt_tmp_J.reshape(b, c2, c2, h1 * w1, h1 * w1)
        Jt_tmp_J = Jt_tmp_J.movedim(2, 3)
        Jt_tmp_J = Jt_tmp_J.reshape(b, c2 * h1 * w1, c2 * h1 * w1)

        return Jt_tmp_J

    def _jacobian_wrt_input_sandwich_full_to_diag(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        raise NotImplementedError

    def _jacobian_wrt_input_sandwich_diag_to_full(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        raise NotImplementedError

    def _jacobian_wrt_input_sandwich_diag_to_diag(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        weight = torch.ones(c2, c1, int(self.scale_factor), int(self.scale_factor), device=x.device)

        tmp_diag = F.conv2d(
            tmp_diag.reshape(-1, c2, h2, w2),
            weight=weight,
            bias=None,
            stride=int(self.scale_factor),
            padding=0,
            dilation=1,
            groups=1,
        )

        return tmp_diag.reshape(b, c1 * h1 * w1)


class Conv1d(AbstractJacobian, nn.Conv1d):
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
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
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
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


def compute_reversed_padding(padding, kernel_size=1):
    return kernel_size - 1 - padding


class Conv2d(AbstractJacobian, nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode
        )

        dw_padding_h = compute_reversed_padding(self.padding[0], kernel_size=self.kernel_size[0])
        dw_padding_w = compute_reversed_padding(self.padding[1], kernel_size=self.kernel_size[1])
        self.dw_padding = (dw_padding_h, dw_padding_w)

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
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

    def _jacobian_wrt_input(self, x: Tensor, val: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        output_identity = torch.eye(c1 * h1 * w1).unsqueeze(0).expand(b, -1, -1)
        output_identity = output_identity.reshape(b, c1, h1, w1, c1 * h1 * w1)

        # convolve each column
        jacobian = self._jacobian_wrt_input_mult_left_vec(x, val, output_identity)

        # reshape as a (num of output)x(num of input) matrix, one for each batch size
        jacobian = jacobian.reshape(b, c2 * h2 * w2, c1 * h1 * w1)

        return jacobian

    def _jacobian_wrt_weight(self, x: Tensor, val: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        kernel_h, kernel_w = self.kernel_size

        output_identity = torch.eye(c2 * c1 * kernel_h * kernel_w)
        # expand rows as [(input channels)x(kernel height)x(kernel width)] cubes, one for each output channel
        output_identity = output_identity.reshape(c2, c1, kernel_h, kernel_w, c2 * c1 * kernel_h * kernel_w)

        reversed_inputs = torch.flip(x, [-2, -1]).movedim(0, 1)

        # convolve each base element and compute the jacobian
        jacobian = (
            F.conv_transpose2d(
                output_identity.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c1, kernel_h, kernel_w),
                weight=reversed_inputs,
                bias=None,
                stride=self.stride,
                padding=self.dw_padding,
                dilation=self.dilation,
                groups=self.groups,
                output_padding=0,
            )
            .reshape(c2, *output_identity.shape[4:], b, h2, w2)
            .movedim((-3, -2, -1), (1, 2, 3))
        )

        # transpose the result in (output height)x(output width)
        jacobian = torch.flip(jacobian, [-3, -2])
        # switch batch size and output channel
        jacobian = jacobian.movedim(0, 1)
        # reshape as a (num of output)x(num of weights) matrix, one for each batch size
        jacobian = jacobian.reshape(b, c2 * h2 * w2, c2 * c1 * kernel_h * kernel_w)
        return jacobian

    def _jacobian_wrt_input_T_mult_right(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        num_of_cols = tmp.shape[-1]
        assert list(tmp.shape) == [b, c2 * h2 * w2, num_of_cols]
        # expand rows as cubes [(output channel)x(output height)x(output width)]
        tmp = tmp.reshape(b, c2, h2, w2, num_of_cols)

        # convolve each column
        Jt_tmp = (
            F.conv_transpose2d(
                tmp.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c2, h2, w2),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                output_padding=self.output_padding,
            )
            .reshape(b, *tmp.shape[4:], c1, h1, w1)
            .movedim((-3, -2, -1), (1, 2, 3))
        )

        # reshape as a (num of input)x(num of column) matrix, one for each batch size
        Jt_tmp = Jt_tmp.reshape(b, c1 * h1 * w1, num_of_cols)
        return Jt_tmp

    def _jacobian_wrt_input_mult_left(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        num_of_rows = tmp.shape[-2]
        assert list(tmp.shape) == [b, num_of_rows, c2 * h2 * w2]
        # expand rows as cubes [(output channel)x(output height)x(output width)]
        tmp_rows = tmp.movedim(-1, -2).reshape(b, c2, h2, w2, num_of_rows)
        # see rows as columns of the transposed matrix
        tmpt_cols = tmp_rows

        # convolve each column
        Jt_tmptt_cols = (
            F.conv_transpose2d(
                tmpt_cols.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c2, h2, w2),
                weight=self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                output_padding=self.output_padding,
            )
            .reshape(b, *tmpt_cols.shape[4:], c1, h1, w1)
            .movedim((-3, -2, -1), (1, 2, 3))
        )

        # reshape as a (num of input)x(num of output) matrix, one for each batch size
        Jt_tmptt_cols = Jt_tmptt_cols.reshape(b, c1 * h1 * w1, num_of_rows)

        # transpose
        tmp_J = Jt_tmptt_cols.movedim(1, 2)
        return tmp_J

    def _jacobian_wrt_weight_T_mult_right(
        self, x: Tensor, val: Tensor, tmp: Tensor, use_less_memory: bool = True
    ) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        kernel_h, kernel_w = self.kernel_size

        num_of_cols = tmp.shape[-1]

        # expand rows as cubes [(output channel)x(output height)x(output width)]
        tmp = tmp.reshape(b, c2, h2, w2, num_of_cols)
        # transpose the images in (output height)x(output width)
        tmp = torch.flip(tmp, [-3, -2])
        # switch batch size and output channel
        tmp = tmp.movedim(0, 1)

        if use_less_memory:
            # define moving sum for Jt_tmp
            Jt_tmp = torch.zeros(b, c2 * c1 * kernel_h * kernel_w, num_of_cols, device=x.device)
            for i in range(b):
                # set the weight to the convolution
                input_single_batch = x[i : i + 1, :, :, :]
                reversed_input_single_batch = torch.flip(input_single_batch, [-2, -1]).movedim(0, 1)

                tmp_single_batch = tmp[:, i : i + 1, :, :, :]

                # convolve each column
                Jt_tmp_single_batch = (
                    F.conv2d(
                        tmp_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, 1, h2, w2),
                        weight=reversed_input_single_batch,
                        bias=None,
                        stride=self.stride,
                        padding=self.dw_padding,
                        dilation=self.dilation,
                        groups=self.groups,
                    )
                    .reshape(c2, *tmp_single_batch.shape[4:], c1, kernel_h, kernel_w)
                    .movedim((-3, -2, -1), (1, 2, 3))
                )

                # reshape as a (num of weights)x(num of column) matrix
                Jt_tmp_single_batch = Jt_tmp_single_batch.reshape(c2 * c1 * kernel_h * kernel_w, num_of_cols)
                Jt_tmp[i, :, :] = Jt_tmp_single_batch

        else:
            reversed_inputs = torch.flip(x, [-2, -1]).movedim(0, 1)

            # convolve each column
            Jt_tmp = (
                F.conv2d(
                    tmp.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, b, h2, w2),
                    weight=reversed_inputs,
                    bias=None,
                    stride=self.stride,
                    padding=self.dw_padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                .reshape(c2, *tmp.shape[4:], c1, kernel_h, kernel_w)
                .movedim((-3, -2, -1), (1, 2, 3))
            )

            # reshape as a (num of weights)x(num of column) matrix
            Jt_tmp = Jt_tmp.reshape(c2 * c1 * kernel_h * kernel_w, num_of_cols)

        return Jt_tmp

    def _jacobian_wrt_weight_mult_left(
        self, x: Tensor, val: Tensor, tmp: Tensor, use_less_memory: bool = True
    ) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]
        kernel_h, kernel_w = self.kernel_size
        num_of_rows = tmp.shape[-2]

        # expand rows as cubes [(output channel)x(output height)x(output width)]
        tmp_rows = tmp.movedim(-1, -2).reshape(b, c2, h2, w2, num_of_rows)
        # see rows as columns of the transposed matrix
        tmpt_cols = tmp_rows
        # transpose the images in (output height)x(output width)
        tmpt_cols = torch.flip(tmpt_cols, [-3, -2])
        # switch batch size and output channel
        tmpt_cols = tmpt_cols.movedim(0, 1)

        if use_less_memory:

            tmp_J = torch.zeros(b, c2 * c1 * kernel_h * kernel_w, num_of_rows, device=x.device)
            for i in range(b):
                # set the weight to the convolution
                input_single_batch = x[i : i + 1, :, :, :]
                reversed_input_single_batch = torch.flip(input_single_batch, [-2, -1]).movedim(0, 1)

                tmp_single_batch = tmpt_cols[:, i : i + 1, :, :, :]

                # convolve each column
                tmp_J_single_batch = (
                    F.conv2d(
                        tmp_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, 1, h2, w2),
                        weight=reversed_input_single_batch,
                        bias=None,
                        stride=self.stride,
                        padding=self.dw_padding,
                        dilation=self.dilation,
                        groups=self.groups,
                    )
                    .reshape(c2, *tmp_single_batch.shape[4:], c1, kernel_h, kernel_w)
                    .movedim((-3, -2, -1), (1, 2, 3))
                )

                # reshape as a (num of weights)x(num of column) matrix
                tmp_J_single_batch = tmp_J_single_batch.reshape(c2 * c1 * kernel_h * kernel_w, num_of_rows)
                tmp_J[i, :, :] = tmp_J_single_batch

            # transpose
            tmp_J = tmp_J.movedim(-1, -2)
        else:
            # set the weight to the convolution
            reversed_inputs = torch.flip(x, [-2, -1]).movedim(0, 1)

            # convolve each column
            Jt_tmptt_cols = (
                F.conv2d(
                    tmp_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, b, h2, w2),
                    weight=reversed_inputs,
                    bias=None,
                    stride=self.stride,
                    padding=self.dw_padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                .reshape(c2, *tmp_single_batch.shape[4:], c1, kernel_h, kernel_w)
                .movedim((-3, -2, -1), (1, 2, 3))
            )

            # reshape as a (num of input)x(num of output) matrix, one for each batch size
            Jt_tmptt_cols = Jt_tmptt_cols.reshape(c2 * c1 * kernel_h * kernel_w, num_of_rows)
            # transpose
            tmp_J = Jt_tmptt_cols.movedim(0, 1)

        return tmp_J

    def _jacobian_wrt_input_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_full(x, val, tmp)
        elif not diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_diag(x, val, tmp)
        elif diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_full(x, val, tmp)
        elif diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, tmp)

    def _jacobian_wrt_weight_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return self._jacobian_wrt_weight_sandwich_full_to_full(x, val, tmp)
        elif not diag_inp and diag_out:
            return self._jacobian_wrt_weight_sandwich_full_to_diag(x, val, tmp)
        elif diag_inp and not diag_out:
            return self._jacobian_wrt_weight_sandwich_diag_to_full(x, val, tmp)
        elif diag_inp and diag_out:
            return self._jacobian_wrt_weight_sandwich_diag_to_diag(x, val, tmp)

    def _jacobian_wrt_input_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        return self._jacobian_wrt_input_mult_left(x, val, self._jacobian_wrt_input_T_mult_right(x, val, tmp))

    def _jacobian_wrt_input_sandwich_full_to_diag(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        raise NotImplementedError

    def _jacobian_wrt_input_sandwich_diag_to_full(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        raise NotImplementedError

    def _jacobian_wrt_input_sandwich_diag_to_diag(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        _, c2, h2, w2 = val.shape

        input_tmp = tmp_diag.reshape(b, c2, h2, w2)

        output_tmp = (
            F.conv_transpose2d(
                input_tmp.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, c2, h2, w2),
                weight=self.weight**2,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                output_padding=0,
            )
            .reshape(b, *input_tmp.shape[4:], c1, h1, w1)
            .movedim((-3, -2, -1), (1, 2, 3))
        )

        diag_Jt_tmp_J = output_tmp.reshape(b, c1 * h1 * w1)
        return diag_Jt_tmp_J

    def _jacobian_wrt_weight_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        return self._jacobian_wrt_weight_mult_left(
            x, val, self._jacobian_wrt_weight_T_mult_right(x, val, tmp)
        )

    def _jacobian_wrt_weight_sandwich_full_to_diag(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        ### TODO: Implement this in a smarter way
        return torch.diagonal(self._jacobian_wrt_weight_sandwich_full_to_full(x, val, tmp), dim1=1, dim2=2)

    def _jacobian_wrt_weight_sandwich_diag_to_full(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        raise NotImplementedError

    def _jacobian_wrt_weight_sandwich_diag_to_diag(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:

        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]
        _, _, kernel_h, kernel_w = self.weight.shape

        input_tmp = tmp_diag.reshape(b, c2, h2, w2)
        # transpose the images in (output height)x(output width)
        input_tmp = torch.flip(input_tmp, [-3, -2, -1])
        # switch batch size and output channel
        input_tmp = input_tmp.movedim(0, 1)

        # define moving sum for Jt_tmp
        output_tmp = torch.zeros(b, c2 * c1 * kernel_h * kernel_w, device=x.device)
        flip_squared_input = torch.flip(x, [-3, -2, -1]).movedim(0, 1) ** 2

        for i in range(b):
            # set the weight to the convolution
            weigth_sq = flip_squared_input[:, i : i + 1, :, :]
            input_tmp_single_batch = input_tmp[:, i : i + 1, :, :]

            output_tmp_single_batch = (
                F.conv2d(
                    input_tmp_single_batch.movedim((1, 2, 3), (-3, -2, -1)).reshape(-1, 1, h2, w2),
                    weight=weigth_sq,
                    bias=None,
                    stride=self.stride,
                    padding=self.dw_padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
                .reshape(c2, *input_tmp_single_batch.shape[4:], c1, kernel_h, kernel_w)
                .movedim((-3, -2, -1), (1, 2, 3))
            )

            output_tmp_single_batch = torch.flip(output_tmp_single_batch, [-4, -3])
            # reshape as a (num of weights)x(num of column) matrix
            output_tmp_single_batch = output_tmp_single_batch.reshape(c2 * c1 * kernel_h * kernel_w)
            output_tmp[i, :] = output_tmp_single_batch

        if self.bias is not None:
            bias_term = tmp_diag.reshape(b, c2, h2*w2)
            bias_term = torch.sum(bias_term, 2)
            output_tmp = torch.cat([output_tmp, bias_term], dim=1)

        return output_tmp


class ConvTranspose2d(AbstractJacobian, nn.ConvTranspose2d):
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
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
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
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
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
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

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        return jac_in.reshape(jac_in.shape[0], *self.dims, *jac_in.shape[2:])

    def _jacobian_wrt_input_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return tmp
        elif not diag_inp and diag_out:
            return torch.diagonal(tmp, dim1=1, dim2=2)
        elif diag_inp and not diag_out:
            return torch.diag_embed(tmp)
        elif diag_inp and diag_out:
            return tmp

    def _jacobian_wrt_weight_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        return None


class Flatten(AbstractJacobian, nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        val = x.reshape(x.shape[0], -1)
        return val

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        if jac_in.ndim == 5:  # 1d conv
            return jac_in.reshape(jac_in.shape[0], -1, *jac_in.shape[3:])
        if jac_in.ndim == 7:  # 2d conv
            return jac_in.reshape(jac_in.shape[0], -1, *jac_in.shape[4:])
        if jac_in.ndim == 9:  # 3d conv
            return jac_in.reshape(jac_in.shape[0], -1, *jac_in.shape[5:])

    def _jacobian_wrt_input_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return tmp
        elif not diag_inp and diag_out:
            return torch.diagonal(tmp, dim1=1, dim2=2)
        elif diag_inp and not diag_out:
            return torch.diag_embed(tmp)
        elif diag_inp and diag_out:
            return tmp

    def _jacobian_wrt_weight_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        return None


class AbstractActivationJacobian:
    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        jac = self._jacobian(x, val)
        n = jac_in.ndim - jac.ndim
        return jac_in * jac.reshape(jac.shape + (1,) * n)

    def __call__(self, x: Tensor, jacobian: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        val = self._call_impl(x)
        if jacobian:
            jac = self._jacobian(x, val)
            return val, jac
        return val


class Softmax(AbstractActivationJacobian, nn.Softmax):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        if self.dim == 0:
            raise ValueError("Jacobian computation not supported for `dim=0`")
        jac = torch.diag_embed(val) - torch.matmul(val.unsqueeze(-1), val.unsqueeze(-2))
        return jac

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        jac = self._jacobian(x, val)
        n = jac_in.ndim - jac.ndim
        jac = jac.reshape((1,) * n + jac.shape)
        if jac_in.ndim == 4:
            return (jac @ jac_in.permute(3, 0, 1, 2)).permute(1, 2, 3, 0)
        if jac_in.ndim == 5:
            return (jac @ jac_in.permute(3, 4, 0, 1, 2)).permute(2, 3, 4, 0, 1)
        if jac_in.ndim == 6:
            return (jac @ jac_in.permute(3, 4, 5, 0, 1, 2)).permute(3, 4, 5, 0, 1, 2)
        return jac @ jac_in


class BatchNorm1d(AbstractActivationJacobian, nn.BatchNorm1d):
    # only implements jacobian during testing
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = (self.weight / (self.running_var + self.eps).sqrt()).unsqueeze(0)
        return jac


class BatchNorm2d(AbstractActivationJacobian, nn.BatchNorm2d):
    # only implements jacobian during testing
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = (self.weight / (self.running_var + self.eps).sqrt()).unsqueeze(0)
        return jac


class BatchNorm3d(AbstractActivationJacobian, nn.BatchNorm3d):
    # only implements jacobian during testing
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = (self.weight / (self.running_var + self.eps).sqrt()).unsqueeze(0)
        return jac


class MaxPool1d(AbstractJacobian, nn.MaxPool1d):
    def forward(self, input: Tensor):
        val, idx = F.max_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            return_indices=True,
        )
        self.idx = idx
        return val

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, l1 = x.shape
        c2, l2 = val.shape[1:]

        jac_in_orig_shape = jac_in.shape
        jac_in = jac_in.reshape(-1, l1, *jac_in_orig_shape[3:])
        arange_repeated = torch.repeat_interleave(torch.arange(b * c1), l2).long()
        idx = self.idx.reshape(-1)
        jac_in = jac_in[arange_repeated, idx, :, :].reshape(*val.shape, *jac_in_orig_shape[3:])
        return jac_in


class MaxPool2d(AbstractJacobian, nn.MaxPool2d):
    def forward(self, input: Tensor):
        val, idx = F.max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            return_indices=True,
        )
        self.idx = idx
        return val

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        jac_in_orig_shape = jac_in.shape
        jac_in = jac_in.reshape(-1, h1 * w1, *jac_in_orig_shape[4:])
        arange_repeated = torch.repeat_interleave(torch.arange(b * c1), h2 * w2).long()
        idx = self.idx.reshape(-1)
        jac_in = jac_in[arange_repeated, idx, :, :, :].reshape(*val.shape, *jac_in_orig_shape[4:])
        return jac_in

    def _jacobian_wrt_weight_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        # non parametric, so return empty
        return None

    def _jacobian_wrt_input_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_full(x, val, tmp)
        elif not diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_diag(x, val, tmp)
        elif diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_full(x, val, tmp)
        elif diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, tmp)

    def _jacobian_wrt_input_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]
        assert c1 == c2

        tmp = tmp.reshape(b, c1, h2 * w2, c1, h2 * w2).movedim(-2, -3).reshape(b * c1 * c1, h2 * w2, h2 * w2)
        Jt_tmp_J = torch.zeros((b * c1 * c1, h1 * w1, h1 * w1), device=tmp.device)
        # indexes for batch and channel
        arange_repeated = torch.repeat_interleave(torch.arange(b * c1 * c1), h2 * w2 * h2 * w2).long()
        arange_repeated = arange_repeated.reshape(b * c1 * c1, h2 * w2, h2 * w2)
        # indexes for height and width
        idx = self.idx.reshape(b, c1, h2 * w2).unsqueeze(2).expand(-1, -1, h2 * w2, -1)
        idx_col = idx.unsqueeze(1).expand(-1, c1, -1, -1, -1).reshape(b * c1 * c1, h2 * w2, h2 * w2)
        idx_row = (
            idx.unsqueeze(2).expand(-1, -1, c1, -1, -1).reshape(b * c1 * c1, h2 * w2, h2 * w2).movedim(-1, -2)
        )

        Jt_tmp_J[arange_repeated, idx_row, idx_col] = tmp
        Jt_tmp_J = (
            Jt_tmp_J.reshape(b, c1, c1, h1 * w1, h1 * w1)
            .movedim(-2, -3)
            .reshape(b, c1 * h1 * w1, c1 * h1 * w1)
        )

        return Jt_tmp_J

    def _jacobian_wrt_input_sandwich_full_to_diag(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        raise NotImplementedError

    def _jacobian_wrt_input_sandwich_diag_to_full(self, x: Tensor, val: Tensor, diag_tmp: Tensor) -> Tensor:
        raise NotImplementedError

    def _jacobian_wrt_input_sandwich_diag_to_diag(self, x: Tensor, val: Tensor, diag_tmp: Tensor) -> Tensor:
        b, c1, h1, w1 = x.shape
        c2, h2, w2 = val.shape[1:]

        new_tmp = torch.zeros_like(x)
        new_tmp = new_tmp.reshape(b * c1, h1 * w1)

        # indexes for batch and channel
        arange_repeated = torch.repeat_interleave(torch.arange(b * c1), h2 * w2).long()
        arange_repeated = arange_repeated.reshape(b * c2, h2 * w2)
        # indexes for height and width
        idx = self.idx.reshape(b * c2, h2 * w2)

        new_tmp[arange_repeated, idx] = diag_tmp.reshape(b * c2, h2 * w2)

        return new_tmp.reshape(b, c1 * h1 * w1)


class MaxPool3d(AbstractJacobian, nn.MaxPool3d):
    def forward(self, input: Tensor):
        val, idx = F.max_pool3d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            return_indices=True,
        )
        self.idx = idx
        return val

    def _jacobian_wrt_input_mult_left_vec(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        b, c1, d1, h1, w1 = x.shape
        c2, d2, h2, w2 = val.shape[1:]

        jac_in_orig_shape = jac_in.shape
        jac_in = jac_in.reshape(-1, d1 * h1 * w1, *jac_in_orig_shape[5:])
        arange_repeated = torch.repeat_interleave(torch.arange(b * c1), h2 * d2 * w2).long()
        idx = self.idx.reshape(-1)
        jac_in = jac_in[arange_repeated, idx, :, :].reshape(*val.shape, *jac_in_orig_shape[5:])
        return jac_in


class Sigmoid(AbstractActivationJacobian, nn.Sigmoid):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = val * (1.0 - val)
        return jac


class ReLU(AbstractActivationJacobian, nn.ReLU):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = (val > 0.0).type(val.dtype)
        return jac


class PReLU(AbstractActivationJacobian, nn.PReLU):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = (val >= 0.0).type(val.dtype) + (val < 0.0).type(val.dtype) * self.weight.reshape(
            (1, self.num_parameters) + (1,) * (val.ndim - 2)
        )
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
        jac = torch.ones_like(val)
        jac[x < 0.0] = self.negative_slope
        return jac


class Softplus(AbstractActivationJacobian, nn.Softplus):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = torch.sigmoid(self.beta * x)
        return jac


class Tanh(AbstractActivationJacobian, nn.Tanh):
    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = 1.0 - val**2
        return jac

    def _jacobian_wrt_weight_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        # non parametric, so return empty
        return None

    def _jacobian_wrt_input_sandwich(
        self, x: Tensor, val: Tensor, tmp: Tensor, diag_inp: bool = False, diag_out: bool = False
    ) -> Tensor:
        if not diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_full(x, val, tmp)
        elif not diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_full_to_diag(x, val, tmp)
        elif diag_inp and not diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_full(x, val, tmp)
        elif diag_inp and diag_out:
            return self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, tmp)

    def _jacobian_wrt_input_sandwich_full_to_full(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        jac = self._jacobian(x, val)
        jac = torch.diag_embed(jac.view(x.shape[0], -1))
        tmp = torch.einsum("bnm,bnj,bjk->bmk", jac, tmp, jac)
        return tmp

    def _jacobian_wrt_input_sandwich_full_to_diag(self, x: Tensor, val: Tensor, tmp: Tensor) -> Tensor:
        jac = self._jacobian(x, val)
        jac = torch.diag_embed(jac.view(x.shape[0], -1))
        tmp = torch.einsum("bnm,bnj,bjm->bm", jac, tmp, jac)
        return tmp

    def _jacobian_wrt_input_sandwich_diag_to_full(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        return torch.diag_embed(self._jacobian_wrt_input_sandwich_diag_to_diag(x, val, tmp_diag))

    def _jacobian_wrt_input_sandwich_diag_to_diag(self, x: Tensor, val: Tensor, tmp_diag: Tensor) -> Tensor:
        jac = self._jacobian(x, val)
        jac = jac.view(x.shape[0], -1)
        tmp = jac**2 * tmp_diag
        return tmp


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
        jac = -1.0 / (x**2 - 1.0)
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
        jac = 0.5 / val
        return jac