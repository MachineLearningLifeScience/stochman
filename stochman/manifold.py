#!/usr/bin/env python3
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.autograd import grad
from torch.distributions import kl_divergence

from stochman.curves import BasicCurve, CubicSpline
from stochman.geodesic import geodesic_minimizing_energy, shooting_geodesic
from stochman.utilities import squared_manifold_distance


class Manifold(ABC):
    """
    A common interface for manifolds. Specific manifolds should inherit
    from this abstract base class abstraction.

    TODO:
    - Add examples to show the differences between Manifold and EmbeddedManifold.
    """

    def curve_energy(self, curve: BasicCurve) -> torch.Tensor:
        """
        Compute the discrete energy of a given curve.

        Input:
            curve:      a Nx(d) torch Tensor representing a curve or
                        a BxNx(d) torch Tensor representing B curves.

        Output:
            energy:     a scalar corresponding to the energy of
                        the curve (sum of energy in case of multiple curves).
                        It should be possible to backpropagate through
                        this in order to compute geodesics.

        Algorithmic note:
            The default implementation of this function rely on the 'inner'
            function, which in turn call the 'metric' function. For some
            manifolds this can be done more efficiently, in which case it
            is recommended that the default implementation is replaced.
        """
        if curve.dim() == 2:
            curve.unsqueeze_(0)  # add batch dimension if one isn't present
        # Now curve is BxNx(d)
        d = curve.shape[2]
        delta = curve[:, 1:] - curve[:, :-1]  # Bx(N-1)x(d)
        flat_delta = delta.view(-1, d)  # (B*(N-1))x(d)
        energy = self.inner(curve[:, :-1].view(-1, d), flat_delta, flat_delta)  # B*(N-1)
        return energy.sum()  # scalar

    def curve_length(self, curve: BasicCurve) -> torch.Tensor:
        """
        Compute the discrete length of a given curve.

        Input:
            curve:      a Nx(d) torch Tensor representing a curve or
                        a BxNx(d) torch Tensor representing B curves.

        Output:
            length:     a scalar or a B element Tensor containing the length of
                        the curve.

        Algorithmic note:
            The default implementation of this function rely on the 'inner'
            function, which in turn call the 'metric' function. For some
            manifolds this can be done more efficiently, in which case it
            is recommended that the default implementation is replaced.
        """
        if curve.dim() == 2:
            curve.unsqueeze_(0)  # add batch dimension if one isn't present
        # Now curve is BxNx(d)
        B, N, d = curve.shape
        delta = curve[:, 1:] - curve[:, :-1]  # Bx(N-1)x(d)
        flat_delta = delta.view(-1, d)  # (B*(N-1))x(d)
        energy = self.inner(curve[:, :-1].view(-1, d), flat_delta, flat_delta)  # B*(N-1)
        length = energy.view(B, N - 1).sqrt().sum(dim=1)  # B
        return length

    @abstractmethod
    def metric(self, points: torch.Tensor) -> torch.Tensor:
        """
        Return the metric tensor at a specified set of points.

        Input:
            points:     a Nx(d) torch Tensor representing a set of
                        points where the metric tensor is to be
                        computed.

        Output:
            M:          a Nx(d)x(d) or Nx(d) torch Tensor representing
                        the metric tensor at the given points.
                        If M is Nx(d)x(d) then M[i] is a (d)x(d) symmetric
                        positive definite matrix. If M is Nx(d) then M[i]
                        is to be interpreted as the diagonal elements of
                        a (d)x(d) diagonal matrix.
        """
        pass

    def inner(
        self, base: torch.Tensor, u: torch.Tensor, v: torch.Tensor, return_metric: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute the inner product between tangent vectors u and v at
        base point.

        Mandatory inputs:
            base:       a Nx(d) torch Tensor representing the points of
                        tangency corresponding to u and v.
            u:          a Nx(d) torch Tensor representing N tangent vectors
                        in the tangent spaces of 'base'.
            v:          a Nx(d) torch Tensor representing tangent vectors.

        Optional input:
            return_metric:  if True, the metric at 'base' is returned as a second
                            output. Otherwise, only one output is provided.

        Output:
            dot:        a N element torch Tensor containing the inner product
                        between u and v according to the metric at base.
            M:          if return_metric=True this second output is also
                        provided. M is a Nx(d)x(d) or a Nx(d) torch Tensor
                        representing the metric tensor at 'base'.
        """
        M = self.metric(base)  # Nx(d)x(d) or Nx(d)
        diagonal_metric = M.dim() == 2
        if diagonal_metric:
            dot = (u * M * v).sum(dim=1)  # N
        else:
            Mv = M.bmm(v.unsqueeze(-1))  # Nx(d)
            dot = u.unsqueeze(1).bmm(Mv).flatten()  # N    #(u * Mv).sum(dim=1) # N
        if return_metric:
            return dot, M
        else:
            return dot

    def volume(self, points: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the volume measure at a set of given points.

        Input:
            points:     a Nx(d) torch Tensor representing points on
                        the manifold.

        Output:
            vol:        a N element torch Tensor containing the volume
                        element at each point.

        Algorithmic note:
            The algorithm merely compute the square root determinant of
            the metric at each point. This may be expensive and may be numerically
            unstable; if possible, you should use the 'log_volume' function
            instead.
        """
        M = self.metric(points)  # Nx(d)x(d) or Nx(d)
        diagonal_metric = M.dim() == 2
        if diagonal_metric:
            vol = M.prod(dim=1).sqrt()  # N
        else:
            vol = M.det().sqrt()  # N
        return vol

    def log_volume(self, points: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the logarithm of the volume measure at a set of given points.

        Input:
            points:     a Nx(d) torch Tensor representing points on
                        the manifold.

        Output:
            log_vol:    a N element torch Tensor containing the logarithm
                        of the volume element at each point.

        Algorithmic note:
            The algorithm merely compute the log-determinant of the metric and
            divide by 2. This may be expensive.
        """
        M = self.metric(points)  # Nx(d)x(d) or Nx(d)
        diagonal_metric = M.dim() == 2
        if diagonal_metric:
            log_vol = 0.5 * M.log().sum(dim=1)  # N
        else:
            log_vol = 0.5 * M.logdet()  # N
        return log_vol

    def geodesic_system(self, c: torch.Tensor, dc: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the geodesic ODE of the manifold.

        Inputs:
            c:          a Nx(d) torch Tensor representing a set of points
                        in latent space (e.g. a curve).
            dc:         a Nx(d) torch Tensor representing the velocity at
                        the points.

        Output:
            ddc:        a Nx(d) torch Tensor representing the second temporal
                        derivative of the geodesic passing through c with
                        velocity dc.

        Algorithmic notes:
            The algorithm evaluates the equation
                c'' = M^{-1} * (0.5 * dL/dc - dM/dt * c')
                L = c'^T * M * c'
            The term dL/dc is evaluated with automatic differentiation, which
            imply a loop over N, which can be slow. The derivative dM/dt is
            evaluated using finite differences, which is fast but may imply
            a slight loss of accuracy.
            When possible, it may be beneficial to provide a specialized version
            of this function.
        """
        N, d = c.shape
        requires_grad = c.requires_grad or dc.requires_grad

        # Compute dL/dc using auto diff
        z = c.clone().requires_grad_()  # Nx(d)
        dz = dc.clone().requires_grad_()  # Nx(d)
        L, M = self.inner(z, dz, dz, return_metric=True)  # N, Nx(d)x(d) or N, Nx(d)
        if requires_grad:
            dLdc = torch.cat([grad(L[n], z, create_graph=True)[0][n].unsqueeze(0) for n in range(N)])  # Nx(d)
        else:
            dLdc = torch.cat(
                [grad(L[n], z, retain_graph=(n < N - 1))[0][n].unsqueeze(0) for n in range(N)]
            )  # Nx(d)

        # Use finite differences to approximate dM/dt as that is more
        # suitable for batching.
        # TODO: make this behavior optional allowing exact expressions.
        # h = 1e-4
        # with torch.set_grad_enabled(requires_grad):
        #    dMdt = (self.metric(z + h*dz) - M) / h # Nx(d)x(d) or Nx(d)
        # print('fd', dMdt, dMdt.shape)

        M = self.metric(z)
        diagonal_metric = M.dim() == 2
        if requires_grad:
            if diagonal_metric:
                dMdt = torch.tensor(
                    [
                        [torch.sum(grad(M[n, i], z, create_graph=True)[0] * dz) for i in range(d)]
                        for n in range(N)
                    ]
                )  # Nx(d)
            else:
                dMdt = torch.tensor(
                    [
                        [
                            torch.sum(grad(M[n, i, j], z, create_graph=True)[0] * dz)
                            for i in range(d)
                            for j in range(d)
                        ]
                        for n in range(N)
                    ]
                ).view(
                    N, d, d
                )  # Nx(d)x(d) # TODO: figure out how to not store the graph
        else:
            if diagonal_metric:
                dMdt = torch.tensor(
                    [
                        [torch.sum(grad(M[n, i], z, retain_graph=True)[0] * dz) for i in range(d)]
                        for n in range(N)
                    ]
                )  # Nx(d) # TODO: figure out how to not store the graph
            else:
                dMdt = torch.tensor(
                    [
                        [
                            torch.sum(grad(M[n, i, j], z, retain_graph=True)[0] * dz)
                            for i in range(d)
                            for j in range(d)
                        ]
                        for n in range(N)
                    ]
                ).view(
                    N, d, d
                )  # Nx(d)x(d) # TODO: figure out how to not store the graph
        # print('ad', dMdt, dMdt.shape)

        # Evaluate full geodesic ODE:
        # c'' = (0.5 * dL/dc - dM/dt * c') / M
        with torch.set_grad_enabled(requires_grad):
            if diagonal_metric:
                ddc = (0.5 * dLdc - dMdt * dz) / M  # Nx(d)
            else:
                # XXX: Consider Cholesky-based solver
                Mddc = 0.5 * dLdc - dMdt.bmm(dz.unsqueeze(-1)).squeeze(-1)  # Nx(d)
                ddc, _ = torch.solve(Mddc.unsqueeze(-1), M)  # Nx(d)x1
                ddc = ddc.squeeze(-1)  # Nx(d)
        return ddc

    def connecting_geodesic(self, p0, p1, init_curve: Optional[BasicCurve] = None) -> Tuple[BasicCurve, bool]:
        """
        Compute geodesic connecting two points.

        Args:
            p0: a torch Tensor representing the initial point of the requested geodesic.
            p1: a torch Tensor representing the end point of the requested geodesic.
            init_curve: a curve representing an initial guess of the requested geodesic.
                If the end-points of the initial curve do not correspond to p0 and p1,
                then the curve is modified accordingly. If None then the default constructor
                of the chosen curve family is applied.
        """
        if init_curve is None:
            curve = CubicSpline(p0, p1)
        else:
            curve = init_curve
            curve.begin = p0
            curve.end = p1

        # success = geodesic_minimizing_energy_sgd(curve, self)
        success = geodesic_minimizing_energy(curve, self)
        return curve, success

    def shooting_geodesic(self, p, v, t=torch.linspace(0, 1, 50), requires_grad=False):
        """
        Compute the geodesic with a given starting point and initial velocity.

        Mandatory inputs:
            p:              a torch Tensor with D elements representing the initial
                            position on the manifold of the requested geodesic.
            v:              a torch Tensor with D elements representing the initial
                            velocity of the requested geodesic.

        Optional inputs:
            t:              a torch Tensor of time values where the requested geodesic
                            will be computed. This must at least contain two values
                            where the first must be 0.
                            Default: torch.linspace(0, 1, 50)
            requires_grad:  if True it is possible to backpropagate through this
                            function.
                            Default: False

        Output:
            c:              a torch Tensor of size TxD containing points along the
                            geodesic at the reequested times.
            dc:             a torch Tensor of size TxD containing the curve derivatives
                            at the requested times.
        """
        return shooting_geodesic(self, p, v, t, requires_grad)

    def logmap(self, p0, p1, curve: Optional[BasicCurve] = None, optimize=True):
        """
        Compute the logarithm map of the geodesic from p0 to p1.

        Mandatory inputs:
            p0:         a torch Tensor representing the base point
                        of the logarithm map.
            p1:         a torch Tensor representing the end point
                        of the underlying geodesic.

        Optional inputs:
            curve:      an initial estimate of the geodesic from
                        p0 to p1.
                        Default: None
            optimize:   if False and an initial curve is present, then
                        the initial curve is assumed to be the true
                        geodesic and the logarithm map is extracted
                        from the initial curve.
                        Default: True

        Output:
            lm:         a torch Tensor with D elements representing
                        a tangent vector at p0. The norm of lm
                        is the geodesic distance from p0 to p1.
        """
        if curve is None:
            curve = self.connecting_geodesic(p0, p1)
        if curve is not None and optimize:
            curve = self.connecting_geodesic(p0, p1, init_curve=curve)
        with torch.no_grad():
            lm = curve.deriv(torch.zeros(1))
        return lm

    def expmap(self, p, v):
        """
        Compute the exponential map starting at p with velocity v.

        Mandatory inputs:
            p:          a torch Tensor representing the base point
                        of the exponential map.
            v:          a torch Tensor representing the velocity of
                        the underlying geodesic at p.

        Output:
            u:          a torch Tensor corresponding to the end-point
                        of the requested geodesic.

        Algorithmic note:
            This implementation use a numerical ODE solver to integrate
            the geodesic ODE. This in turn require evaluating both the
            metric and its derivatives, which may be expensive.
        """
        requires_grad = p.requires_grad or v.requires_grad
        c, _ = shooting_geodesic(self, p, v, t=torch.linspace(0, 1, 5), requires_grad=requires_grad)
        return c[-1].view(1, -1)

    def dist2(self, p0: torch.Tensor, p1: torch.Tensor):
        """
        Compute the squared geodesic distance between two points.

        Mandatory inputs:
            p0: a torch Tensor representing one point.
            p1: a torch Tensor representing another point.

        Output:
            d2: the squared geodesic distance between the two
                given points.
        """
        # TODO: allow for warm-starting the geodesic
        return squared_manifold_distance(self, p0, p1)


class EmbeddedManifold(Manifold, ABC):
    """
    A common interface for embedded manifolds. Specific embedded manifolds
    should inherit from this abstract base class abstraction.
    """

    def curve_energy(self, curve: BasicCurve, dt=None):
        """
        Compute the discrete energy of a given curve.

        Input:
            curve:      a Nx(d) torch Tensor representing a curve or
                        a BxNx(d) torch Tensor representing B curves.

        Output:
            energy:     a scalar corresponding to the energy of
                        the curve (sum of energy in case of multiple curves).
                        It should be possible to backpropagate through
                        this in order to compute geodesics.

        Algorithmic note:
            The algorithm rely on the deterministic embedding of the manifold
            rather than the metric. This is most often more efficient.
        """
        if curve.dim() == 2:
            curve.unsqueeze_(0)  # add batch dimension if one isn't present
        if dt is None:
            dt = curve.shape[1] - 1
        # Now curve is BxNx(d)
        emb_curve = self.embed(curve)  # BxNxD
        B, N, D = emb_curve.shape
        delta = emb_curve[:, 1:, :] - emb_curve[:, :-1, :]  # Bx(N-1)xD
        energy = (delta ** 2).sum((1, 2)) * dt  # B

        return energy

    def curve_length(self, curve: BasicCurve, dt=None):
        """
        Compute the discrete length of a given curve.

        Input:
            curve:      a Nx(d) torch Tensor representing a curve or
                        a BxNx(d) torch Tensor representing B curves.

        Output:
            length:     a scalar or a B element Tensor containing the length of
                        the curve.

        Algorithmic note:
            The default implementation of this function rely on the 'inner'
            function, which in turn call the 'metric' function. For some
            manifolds this can be done more efficiently, in which case it
            is recommended that the default implementation is replaced.
        """
        if curve.dim() == 2:
            curve.unsqueeze_(0)  # add batch dimension if one isn't present
        if dt is None:
            dt = 1.0  # (curve.shape[1]-1)
        # Now curve is BxNx(d)
        emb_curve = self.embed(curve)  # BxNxD
        delta = emb_curve[:, 1:] - emb_curve[:, :-1]  # Bx(N-1)xD
        speed = delta.norm(dim=2)  # Bx(N-1)
        lengths = speed.sum(dim=1) * dt  # B
        return lengths

    def metric(self, points: torch.Tensor):
        """
        Return the metric tensor at a specified set of points.

        Input:
            points:     a Nx(d) torch Tensor representing a set of
                        points where the metric tensor is to be
                        computed.

        Output:
            M:          a Nx(d)x(d) or Nx(d) torch Tensor representing
                        the metric tensor at the given points.
                        If M is Nx(d)x(d) then M[i] is a (d)x(d) symmetric
                        positive definite matrix. If M is Nx(d) then M[i]
                        is to be interpreted as the diagonal elements of
                        a (d)x(d) diagonal matrix.
        """
        _, J = self.embed(points, jacobian=True)  # NxDx(d)
        M = torch.einsum("bji,bjk->bik", J, J)
        return M

    @abstractmethod
    def embed(self, points: torch.Tensor, jacobian: bool = False):
        """
        XXX: Write me! Don't forget batching!
        """
        pass


class LocalVarMetric(Manifold):
    r"""
    A class for computing the local inverse-variance metric described in

        A Locally Adaptive Normal Distribution
        Georgios Arvanitidis, Lars Kai Hansen, and Søren Hauberg.
        Neural Information Processing Systems, 2016.

    along with associated quantities. The metric is diagonal with elements
    corresponding to the inverse (reciprocal) of the local variance, which
    is defined as

        var(x) = \sum_n w_n(x) * (x_n - x)^2 + rho
        w_n(x) = exp(-|x_n - x| / 2sigma^2)
    """

    def __init__(self, data, sigma, rho, device=None):
        """
        Class constructor.

        Mandatory inputs:
            data:   The data from which local variance will be computed.
                    This should be a NxD torch Tensor correspondong to
                    N observations of dimension D.
            sigma:  The width of the Gaussian window used to define
                    locality when computing weights. This must be a positive
                    scalar.
            rho:    The added bias in the local variance estimate. This
                    must be a positive scalar (usually a small number, e.g. 1e-4).

        Optional inputs:
            device: The torch device on which computations will be performed.
                    Default: None
        """
        super().__init__()
        self.data = data
        self.sigma2 = sigma ** 2
        self.rho = rho
        self.device = device

    def metric(self, c, return_deriv=False):
        """
        Evaluate the local inverse-variance metric tensor at a given set of points.

        Mandatory input:
          c:              A PxD torch Tensor containing P points of dimension D where
                          the metric will be evaluated.

        Optional input:
          return_deriv:   If True the function will return a second output containing
                          the derivative of the metric tensor. This will be returned
                          in the form of a PxDxD torch Tensor.
                          Default: False

        Output:
          M:              The diagonal elements of the inverse-variance metric
                          represented as a PxD torch Tensor.
        """
        X = self.data  # NxD
        N = X.shape[0]
        if c.ndim == 1:
            c = c.view(1, -1)
        P, D = c.shape
        sigma2 = self.sigma2
        rho = self.rho
        K = 1.0 / ((2.0 * np.pi * sigma2) ** (D / 2.0))

        # Compute metric
        M = []  # torch.empty((P, D)) # metric
        dMdc = []  # derivative of metric in case it is requested
        for p in range(P):
            delta = X - c[p]  # NxD
            delta2 = (delta) ** 2  # NxD
            dist2 = delta2.sum(dim=1)  # N
            w_p = K * torch.exp(-0.5 * dist2 / sigma2).reshape((1, N))  # 1xN
            S = w_p.mm(delta2) + rho  # D
            m = 1.0 / S  # D
            M.append(m)
            if return_deriv:
                weighted_delta = (w_p / sigma2).reshape(-1, 1).expand(-1, D) * delta  # NxD
                dSdc = 2.0 * torch.diag(w_p.mm(delta).flatten()) - weighted_delta.t().mm(delta2)  # DxD
                dM = dSdc.t() * (m ** 2).reshape(-1, 1).expand(-1, D)  # DxD
                dMdc.append(dM.reshape(1, D, D))

        if return_deriv:
            return torch.cat(M), torch.cat(dMdc, dim=0)
        else:
            return torch.cat(M)

    def curve_energy(self, c):
        """
        Evaluate the energy of a curve represented as a discrete set of points.

        Input:
            c:      A discrete set of points along a curve. This is represented
                    as a PxD or BxPxD torch Tensor. The points are assumed to be ordered
                    along the curve and evaluated at equidistant time points.

        Output:
            energy: The energy of the input curve.
        """
        if len(c.shape) == 2:
            c.unsqueeze_(0)  # add batch dimension if one isn't present
        energy = torch.zeros(1)
        for b in range(c.shape[0]):
            M = self.metric(c[b, :-1])  # (P-1)xD
            delta1 = (c[b, 1:] - c[b, :-1]) ** 2  # (P-1)xD
            energy += (M * delta1).sum()
        return energy

    def curve_length(self, c):
        """
        Evaluate the length of a curve represented as a discrete set of points.

        Input:
            c:      A discrete set of points along a curve. This is represented
                    as a PxD torch Tensor. The points are assumed to be ordered
                    along the curve and evaluated at equidistant indices.

        Output:
            length: The length of the input curve.
        """
        M = self.metric(c[:-1])  # (P-1)xD
        delta1 = (c[1:] - c[:-1]) ** 2  # (P-1)xD
        length = (M * delta1).sum(dim=1).sqrt().sum()
        return length

    def geodesic_system(self, c, dc):
        """
        Evaluate the 2nd order system of ordinary differential equations that
        govern geodesics.

        Inputs:
            c:      A NxD torch Tensor of D-dimensional points on the manifold.
            dc:     A NxD torch Tensor of first derivatives at the points specified
                    by the first input argument.

        Output:
            ddc:    A NxD torch Tensor of second derivatives at the specified locations.
        """
        N, D = c.shape
        M, dM = self.metric(c, return_deriv=True)  # [NxD, NxDxD]

        # Prepare the output
        ddc = []  # torch.zeros(D, N) # DxN

        # Evaluate the geodesic system
        for n in range(N):
            dMn = dM[n]  # DxD
            ddc_n = (
                -0.5
                * (2.0 * (dMn * dc[n].reshape(-1, 1).expand(-1, D)).mv(dc[n]) - dMn.t().mv(dc[n] ** 2))
                / M[n].flatten()
            )
            ddc.append(ddc_n.reshape(D, 1))

        ddc_tensor = torch.cat(ddc, dim=1).t()  # NxD
        return ddc_tensor


class StochasticManifold(Manifold):
    """
    A class for computing Stochastic Manifolds and defining
    a geometry in the latent space of a certain model
    using the pullback of the Fisher-Rao metric.

    The methods for computing shortest paths are layed out
    in
        Pulling Back Information Geometry
        TODO: Add authors and the rest of the info.

    TODO:
        -  Right now we are getting geodesics by computing shortest paths.What about
           computing the metric? The code we have for approximating the FR
           metric is flimsy. Søren had the idea of having a `kl_divergence`-like
           interface for computing FRs, and then using Jacobians to get an approximation
           of the metric itself.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Class constructor:

        Arguments:
        - model: a torch module that implements a `decode(z: Tensor) -> Distribution`.

        TODO:
            - Should we inherit from EmbeddedManifold and use the `embed` function instead?
        """
        super().__init__()

        self.model = model
        assert "decode" in dir(model)

    def curve_energy(self, curve: BasicCurve) -> torch.Tensor:
        dt = (curve[:-1] - curve[1:]).pow(2).sum(dim=-1, keepdim=True)  # (N-1)x1
        dist1 = self.model.decode(curve[:-1])
        dist2 = self.model.decode(curve[1:])

        try:
            kl = kl_divergence(dist1, dist2)
        except Exception:
            # TODO: fix the exception.
            raise ValueError("Did you forget to register your KL?")

        return kl.sum() * (2 * (dt.mean() ** -1))

    def curve_length(self, curve: BasicCurve) -> torch.Tensor:
        raise NotImplementedError

    def metric(self, points: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
