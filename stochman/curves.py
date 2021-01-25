#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn


class BasicCurve(ABC, nn.Module):
    def __init__(
        self,
        begin: torch.Tensor,
        end: torch.Tensor,
        num_nodes: int = 5,
        requires_grad: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self._begin = begin
        self._end = end
        self._num_nodes = num_nodes
        self._requires_grad = requires_grad

        # register begin and end as buffers
        if len(self._begin.shape) == 1 or self._begin.shape[0] == 1:
            self.register_buffer("begin", self._begin.detach().view((1, -1)))  # 1xD
        else:
            self.register_buffer("begin", self._begin.detach())  # BxD

        if len(self._end.shape) == 1 or self._end.shape[0] == 1:
            self.register_buffer("end", self._end.detach().view((1, -1)))  # 1xD
        else:
            self.register_buffer("end", self._end.detach())  # BxD

        # overriden by child modules
        self._init_params(*args, **kwargs)

    @abstractmethod
    def _init_params(self, *args, **kwargs) -> None:
        pass

    @property
    def device(self):
        """ Returns the device of the curve. """
        return self.params.device

    def plot(self, t0: float = 0.0, t1: float = 1.0, N: int = 100, *plot_args, **plot_kwargs):
        """Plot the curve.

        Args:
            t0: initial timepoint
            t1: final timepoint
            N: number of points used for plotting the curve
            plot_args: additional arguments passed directly to plt.plot
            plot_kwargs: additional keyword-arguments passed directly to plt.plot

        Returns:
            figs: figure handles

        """
        with torch.no_grad():
            import torchplot as plt

            t = torch.linspace(t0, t1, N, dtype=self.begin.dtype, device=self.device)
            points = self(t)  # NxD or BxNxD

            if len(points.shape) == 2:
                points.unsqueeze_(0)  # 1xNxD

            figs = []
            if points.shape[-1] == 1:
                for b in range(points.shape[0]):
                    fig = plt.plot(t, points[b], *plot_args, **plot_kwargs)
                    figs.append(fig)
                return figs
            if points.shape[-1] == 2:
                for b in range(points.shape[0]):
                    fig = plt.plot(points[b, :, 0], points[b, :, 1], "-", *plot_args, **plot_kwargs)
                    figs.append(fig)
                return figs

            raise ValueError(
                "BasicCurve.plot only supports plotting curves in"
                f" 1D or 2D, but recieved points with shape {points.shape}"
            )

    def euclidean_length(self, t0: float = 0.0, t1: float = 1.0, N: int = 100) -> torch.Tensor:
        """Calculate the euclidian length of the curve
        Args:
            t0: starting time
            t1: end time
            N: number of discretized points

        Returns:
            lengths: a tensor with the length of each curve
        """
        t = torch.linspace(t0, t1, N, device=self.device)
        points = self(t)  # NxD or BxNxD
        is_batched = points.dim() > 2
        if not is_batched:
            points = points.unsqueeze(0)
        delta = points[:, 1:] - points[:, :-1]  # Bx(N-1)xD
        energies = (delta ** 2).sum(dim=2)  # Bx(N-1)
        lengths = energies.sqrt().sum(dim=1)  # B
        return lengths

    def fit(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        num_steps: int = 50,
        threshold: float = 1e-6,
        **optimizer_kwargs,
    ) -> torch.Tensor:
        """Fit the curve to the points by minimizing |x - c(t)|²

        Args:
            t:  a torch tensor with N elements showing where to evaluate the curve.
            x:  a torch tensor of size NxD containing the requested
                values the curve should take at time t.
            num_steps: number of optimization steps
            threshold: stopping criterium
            optimizer_kwargs: additional keyword arguments (like lr) passed to the optimizer

        Returns:
            loss: optimized loss

        """
        # using a second order method on a linear problem should imply
        # that we get to the optimum in few iterations (ideally 1).
        opt = torch.optim.LBFGS(self.parameters(), **optimizer_kwargs)
        loss_func = torch.nn.MSELoss()

        def closure():
            opt.zero_grad()
            L = loss_func(self(t), x)
            L.backward()
            return L

        for _ in range(num_steps):
            loss = opt.step(closure=closure)
            if torch.max(torch.abs(self.params.grad)) < threshold:
                break
        return loss


class DiscreteCurve(BasicCurve):
    def __init__(
        self, begin: torch.Tensor, end: torch.Tensor, num_nodes: int = 5, requires_grad: bool = True
    ) -> None:
        super().__init__(begin, end, num_nodes, requires_grad)

    def _init_params(self, *args, **kwargs) -> None:
        self.register_buffer(
            "t",
            torch.linspace(0, 1, self._num_nodes, dtype=self._begin.dtype)[1:-1]
            .reshape(-1, 1, 1)
            .repeat(1, *self.begin.shape),
        )
        params = self.t * self.end.unsqueeze(0) + (1 - self.t) * self.begin.unsqueeze(0)
        if self._requires_grad:
            self.register_parameter("params", nn.Parameter(params))
        else:
            self.register_buffer("params", params)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        start_nodes = torch.cat((self.begin.unsqueeze(0), self.params))  # (num_edges)xBxD
        end_nodes = torch.cat((self.params, self.end.unsqueeze(0)))  # (num_edges)xBxD
        num_edges, B, D = start_nodes.shape
        t0 = torch.cat(
            (
                torch.zeros(1, B, D, dtype=self.t.dtype, device=self.device),
                self.t,
                torch.ones(1, B, D, dtype=self.t.dtype, device=self.device),
            )
        )
        a = (end_nodes - start_nodes) / (t0[1:] - t0[:-1])  # (num_edges)xBxD
        b = start_nodes - a * t0[:-1]  # (num_edges)xBxD

        idx = (
            torch.floor(t.flatten() * num_edges).clamp(min=0, max=num_edges - 1).long()
        )  # use this if nodes are equi-distant
        tt = t.view((-1, 1, 1)).expand(-1, B, D)
        result = a[idx] * tt + b[idx]  # (num_edges)xBxD
        return result.permute(1, 0, 2).squeeze(0)  # Bx(num_edges)xD


class CubicSpline(BasicCurve):
    def __init__(
        self,
        begin: torch.Tensor,
        end: torch.Tensor,
        num_nodes: int = 5,
        requires_grad: bool = True,
        basis: Optional[torch.Tensor] = None,
        params: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(begin, end, num_nodes, requires_grad, basis=basis, params=params)

    def _init_params(self, basis, params) -> None:
        if basis is None:
            basis = self.compute_basis(num_edges=self._num_nodes - 1)
        self.register_buffer("basis", basis)

        if params is None:
            params = torch.zeros(
                self.begin.shape[0], self.basis.shape[1], self.begin.shape[1], dtype=self.begin.dtype
            )
        else:
            params = params.unsqueeze(0) if params.ndim == 2 else params

        if self._requires_grad:
            self.register_parameter("params", nn.Parameter(params))
        else:
            self.register_buffer("params", params)

    # Compute cubic spline basis with end-points (0, 0) and (1, 0)
    def compute_basis(self, num_edges) -> torch.Tensor:
        with torch.no_grad():
            # set up constraints
            t = torch.linspace(0, 1, num_edges + 1, dtype=self.begin.dtype)[1:-1]

            end_points = torch.zeros(2, 4 * num_edges, dtype=self.begin.dtype)
            end_points[0, 0] = 1.0
            end_points[1, -4:] = 1.0

            zeroth = torch.zeros(num_edges - 1, 4 * num_edges, dtype=self.begin.dtype)
            for i in range(num_edges - 1):
                si = 4 * i  # start index
                fill = torch.tensor([1.0, t[i], t[i] ** 2, t[i] ** 3], dtype=self.begin.dtype)
                zeroth[i, si : (si + 4)] = fill
                zeroth[i, (si + 4) : (si + 8)] = -fill

            first = torch.zeros(num_edges - 1, 4 * num_edges, dtype=self.begin.dtype)
            for i in range(num_edges - 1):
                si = 4 * i  # start index
                fill = torch.tensor([0.0, 1.0, 2.0 * t[i], 3.0 * t[i] ** 2], dtype=self.begin.dtype)
                first[i, si : (si + 4)] = fill
                first[i, (si + 4) : (si + 8)] = -fill

            second = torch.zeros(num_edges - 1, 4 * num_edges, dtype=self.begin.dtype)
            for i in range(num_edges - 1):
                si = 4 * i  # start index
                fill = torch.tensor([0.0, 0.0, 6.0 * t[i], 2.0], dtype=self.begin.dtype)
                second[i, si : (si + 4)] = fill
                second[i, (si + 4) : (si + 8)] = -fill

            constraints = torch.cat((end_points, zeroth, first, second))
            self.constraints = constraints

            ## Compute null space, which forms our basis
            _, S, V = torch.svd(constraints, some=False)
            basis = V[:, S.numel() :]  # (num_coeffs)x(intr_dim)

            return basis

    def get_coeffs(self) -> torch.Tensor:
        coeffs = (
            self.basis.unsqueeze(0).expand(self.params.shape[0], -1, -1).bmm(self.params)
        )  # Bx(num_coeffs)xD
        B, num_coeffs, D = coeffs.shape
        degree = 4
        num_edges = num_coeffs // degree
        coeffs = coeffs.view(B, num_edges, degree, D)  # Bx(num_edges)x4xD
        return coeffs

    def _eval_polynomials(self, t: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        # each row of coeffs should be of the form c0, c1, c2, ... representing polynomials
        # of the form c0 + c1*t + c2*t^2 + ...
        # coeffs: Bx(num_edges)x(degree)xD
        B, num_edges, degree, D = coeffs.shape
        idx = torch.floor(t * num_edges).clamp(min=0, max=num_edges - 1).long()  # Bx|t|
        power = (
            torch.arange(0.0, degree, dtype=t.dtype, device=self.device).view(1, 1, -1).expand(B, -1, -1)
        )  # Bx1x(degree)
        tpow = t.view(B, -1, 1).pow(power)  # Bx|t|x(degree)
        coeffs_idx = torch.cat([coeffs[k, idx[k]].unsqueeze(0) for k in range(B)])  # Bx|t|x(degree)xD
        retval = torch.sum(tpow.unsqueeze(-1).expand(-1, -1, -1, D) * coeffs_idx, dim=2)  # Bx|t|xD
        return retval

    def _eval_straight_line(self, t: torch.Tensor) -> torch.Tensor:
        B, T = t.shape
        tt = t.view(B, T, 1)  # Bx|t|x1
        retval = (1 - tt).bmm(self.begin.unsqueeze(1)) + tt.bmm(self.end.unsqueeze(1))  # Bx|t|xD
        return retval

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        coeffs = self.get_coeffs()  # Bx(num_edges)x4xD
        no_batch = t.ndim == 1
        if no_batch:
            t = t.expand(coeffs.shape[0], -1)  # Bx|t|
        retval = self._eval_polynomials(t, coeffs)  # Bx|t|xD
        # tt = t.view((-1, 1)).unsqueeze(0).expand(retval.shape[0], -1, -1) # Bx|t|x1
        # retval += (1-tt).bmm(self.begin.unsqueeze(1)) + tt.bmm(self.end.unsqueeze(1)) # Bx|t|xD
        retval += self._eval_straight_line(t)
        if no_batch and retval.shape[0] == 1:
            retval.squeeze_(0)  # |t|xD
        return retval

    def __getitem__(self, indices: int) -> "CubicSpline":
        C = CubicSpline(
            begin=self.begin[indices],
            end=self.end[indices],
            num_nodes=self._num_nodes,
            requires_grad=self._requires_grad,
            basis=self.basis,
            parameters=self.parameters[indices],
        ).to(self.device)
        return C

    def __setitem__(self, indices, curves) -> None:
        self.parameters[indices] = curves.params

    def deriv(self, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return the derivative of the curve at a given time point.
        """
        coeffs = self.get_coeffs()  # Bx(num_edges)x4xD
        B, num_edges, degree, D = coeffs.shape
        dcoeffs = coeffs[:, :, 1:, :] * torch.arange(
            1.0, degree, dtype=coeffs.dtype, device=self.device
        ).view(1, 1, -1, 1).expand(
            B, num_edges, -1, D
        )  # Bx(num_edges)x3xD
        delta = self.end - self.begin  # BxD
        if t is None:
            # construct the derivative spline
            print("WARNING: Construction of spline derivative objects is currently broken!")
            Z = torch.zeros(B, num_edges, 1, D)  # Bx(num_edges)x1xD
            new_coeffs = torch.cat((dcoeffs, Z), dim=2)  # Bx(num_edges)x4xD
            print("***", new_coeffs[0, 0, :, 0])
            retval = CubicSpline(begin=delta, end=delta, num_nodes=self.num_nodes)
            retval.parameters = (
                retval.basis.t().expand(B, -1, -1).bmm(new_coeffs.view(B, -1, D))
            )  # Bx|parameters|xD
        else:
            if t.dim() == 1:
                t = t.expand(coeffs.shape[0], -1)  # Bx|t|
            # evaluate the derivative spline
            retval = self.__ppeval__(t, dcoeffs)  # Bx|t|xD
            # tt = t.view((-1, 1)) # |t|x1
            retval += delta.unsqueeze(1)
        return retval

    def constant_speed(
        self, metric=None, t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reparametrize the curve to have constant speed.

        Optional input:
            metric:     the Manifold under which the curve should have constant speed.
                        If None then the Euclidean metric is applied.
                        Default: None.

        Note: It is not possible to back-propagate through this function.
        Note: This function does currently not support batching.
        """
        with torch.no_grad():
            if t is None:
                t = torch.linspace(0, 1, 100)  # N
            Ct = self(t)  # NxD or BxNxD
            if Ct.dim() == 2:
                Ct.unsqueeze_(0)  # BxNxD
            B, N, D = Ct.shape
            delta = Ct[:, 1:] - Ct[:, :-1]  # Bx(N-1)xD
            if metric is None:
                local_len = delta.norm(dim=2)  # Bx(N-1)
            else:
                local_len = (
                    metric.inner(Ct[:, :-1].reshape(-1, D), delta.view(-1, D), delta.view(-1, D))
                    .view(B, N - 1)
                    .sqrt()
                )  # Bx(N-1)
            cs = local_len.cumsum(dim=1)  # Bx(N-1)
            new_t = torch.cat((torch.zeros(B, 1), cs / cs[:, -1].unsqueeze(1)), dim=1)  # BxN
            with torch.enable_grad():
                _ = self.fit(new_t, Ct)
            return new_t, Ct
