#!/usr/bin/env python3
from math import ceil
from typing import Optional, Tuple, Union

import networkx as nx
import torch

from stochman.curves import CubicSpline, DiscreteCurve
from stochman.manifold import Manifold


class DiscretizedManifold(Manifold):
    def __init__(self):
        self.grid = []
        self.grid_size = []
        self.G = nx.Graph()
        self.__metric__ = torch.Tensor()
        self._diagonal_metric = False
        self._alpha = torch.Tensor()

    def fit(self, model, grid, use_diagonals=True, batch_size=4, interpolation_noise=0.0):
        """
        Discretize a manifold to a given grid.

        Input:
            model:      a stochman.Manifold that is to be approximed with a graph.

            grid:       a list of torch.linspace's that defines the grid over which
                        the manifold will be discretized. For example,
                        grid = [torch.linspace(-3, 3, 50), torch.linspace(-3, 3, 50)]
                        will discretize a two-dimensional manifold on a 50x50 grid.

            use_diagonals:
                        If True, diagonal edges are included in the graph, otherwise
                        they are excluded.
                        Default: True.

            batch_size: Number of edge-lengths that are computed in parallel. The larger
                        value you pick here, the faster the discretization will be.
                        However, memory usage increases with this number, so a good
                        choice is model and hardware specific.
                        Default: 4.

            interpolation_noise:
                        On fitting, the manifold metric is evalated on the provided grid.
                        The `metric` function then performs interpolation of this metric,
                        using the mean of a Gaussian process. The observation noise of
                        this GP regressor can be tuned through the `interpolation_noise`
                        argument.
                        Default: 0.0.
        """
        self.grid = grid
        self.grid_size = [g.numel() for g in grid]
        self.G = nx.Graph()

        dim = len(grid)
        if len(grid) != 2:
            raise Exception('Currently we only support 2D grids -- sorry!')

        # Add nodes to graph
        xsize, ysize = len(grid[0]), len(grid[1])
        node_idx = lambda x, y: x * ysize + y
        self.G.add_nodes_from(range(xsize * ysize))

        point_set = torch.cartesian_prod(
            torch.linspace(0, xsize - 1, xsize, dtype=torch.long),
            torch.linspace(0, ysize - 1, ysize, dtype=torch.long)
        )  # (big)x2

        point_sets = []  # these will be [N, 2] matrices of index points
        neighbour_funcs = []  # these will be functions for getting the neighbour index

        # add sets
        point_sets.append(point_set[point_set[:, 0] > 0])  # x > 0
        neighbour_funcs.append([lambda x: x - 1, lambda y: y])

        point_sets.append(point_set[point_set[:, 1] > 0])  # y > 0
        neighbour_funcs.append([lambda x: x, lambda y: y - 1])

        point_sets.append(point_set[point_set[:, 0] < xsize - 1])  # x < xsize-1
        neighbour_funcs.append([lambda x: x + 1, lambda y: y])

        point_sets.append(point_set[point_set[:, 1] < ysize - 1])  # y < ysize-1
        neighbour_funcs.append([lambda x: x, lambda y: y + 1])

        if use_diagonals:
            point_sets.append(point_set[torch.logical_and(point_set[:, 0] > 0, point_set[:, 1] > 0)])
            neighbour_funcs.append([lambda x: x - 1, lambda y: y - 1])

            point_sets.append(point_set[torch.logical_and(point_set[:, 0] < xsize - 1, point_set[:, 1] > 0)])
            neighbour_funcs.append([lambda x: x + 1, lambda y: y - 1])

        t = torch.linspace(0, 1, 2)
        for ps, nf in zip(point_sets, neighbour_funcs):
            for i in range(ceil(ps.shape[0] / batch_size)):
                x = ps[batch_size * i:batch_size * (i + 1), 0]
                y = ps[batch_size * i:batch_size * (i + 1), 1]
                xn, yn = nf[0](x), nf[1](y)

                bs = x.shape[0]  # may be different from batch size for the last batch

                line = CubicSpline(begin=torch.zeros(bs, dim), end=torch.ones(bs, dim), num_nodes=2)
                line.begin = torch.cat([grid[0][x].view(-1, 1), grid[1][y].view(-1, 1)], dim=1)  # (bs)x2
                line.end = torch.cat([grid[0][xn].view(-1, 1), grid[1][yn].view(-1, 1)], dim=1)  # (bs)x2

                # if external_curve_length_function:
                #     weight = external_curve_length_function(model, line(t))
                # else:
                with torch.no_grad():
                    weight = model.curve_length(line(t))

                node_index1 = node_idx(x, y)
                node_index2 = node_idx(xn, yn)

                for n1, n2, w in zip(node_index1, node_index2, weight):
                    self.G.add_edge(n1.item(), n2.item(), weight=w.item())

        # Evaluate metric at grid
        try:
            Mlist = []
            with torch.no_grad():
                for x in range(xsize):
                    for y in range(ysize):
                        p = torch.tensor([self.grid[0][x], self.grid[1][y]])
                        Mlist.append(model.metric(p))  # 1x(d)x(d) or 1x(d)
            M = torch.cat(Mlist, dim=0)  # (big)x(d)x(d) or (big)x(d)
            self._diagonal_metric = M.dim() == 2
            d = M.shape[-1]
            if self._diagonal_metric:
                self.__metric__ = M.view([*self.grid_size, d])  # e.g. (xsize)x(ysize)x(d)
            else:
                self.__metric__ = M.view([*self.grid_size, d, d])  # e.g. (xsize)x(ysize)x(d)x(d)

            # Compute interpolation weights. We use the mean function of a GP regressor.
            mesh = torch.meshgrid(*self.grid, indexing='ij')
            grid_points = torch.cat(
                [m.unsqueeze(-1) for m in mesh], dim=-1
            )  # e.g. 100x100x2 a 2D grid with 100 points in each dim
            K = self._kernel(grid_points.view(-1, len(self.grid)))  # (num_grid)x(num_grid)
            if interpolation_noise > 0.0:
                K += interpolation_noise * torch.eye(K.shape[0])
            num_grid = K.shape[0]
            self._alpha = torch.linalg.solve(
                K, self.__metric__.view(num_grid, -1)
            )  # (num_grid)x(d²) or (num_grid)x(d)
        except:
            import warnings
            warnings.warn("It appears that your model does not implement a metric.")
            # XXX: Down the road, we should be able to estimate the metric from the observed distances

    # def set(self, metric, grid, use_diagonals=True):
    #     """
    #     be able to set metric directly from a pre-evaluated grid -- tihs is currently not implemented
    #     """
    #     pass

    def metric(self, points):
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
        # XXX: We should also support returning the derivative of the metric! (for ODEs; see local_PCA)
        K = self._kernel(points)  # Nx(num_grid)
        M = K.mm(self._alpha)  # Nx(d²) or Nx(d)
        if not self._diagonal_metric:
            d = len(self.grid)
            M = M.view(-1, d, d)
        return M

    def _grid_dist2(self, p):
        """Return the squared Euclidean distance from a set of points to the grid.

        Input:
            p:      a Nx(d) torch Tensor corresponding to N latent points.

        Output:
            dist2:  a NxM torch Tensor containing all Euclidean distances
                    to the M grid points.
        """

        dist2 = torch.zeros(p.shape[0], self.G.number_of_nodes())
        mesh = torch.meshgrid(*self.grid, indexing='ij')
        for mesh_dim, dim in zip(mesh, range(len(self.grid))):
            dist2 += (p[:, dim].view(-1, 1) - mesh_dim.reshape(1, -1))**2
        return dist2

    def _kernel(self, p):
        """Evaluate the interpolation kernel for computing the metric.

        Input:
            p:      a torch Tensor corresponding to a point on the manifold.

        Output:
            val:    a torch Tensor with the kernel values.
        """
        lengthscales = [(g[1] - g[0])**2 for g in self.grid]

        dist2 = torch.zeros(p.shape[0], self.G.number_of_nodes())
        mesh = torch.meshgrid(*self.grid, indexing='ij')
        for mesh_dim, dim in zip(mesh, range(len(self.grid))):
            dist2 += (p[:, dim].view(-1, 1) - mesh_dim.reshape(1, -1))**2 / lengthscales[dim]

        return torch.exp(-dist2)

    def _grid_point(self, p):
        """Return the index of the nearest grid point.

        Input:
            p:      a torch Tensor corresponding to a latent point.

        Output:
            idx:    an integer correponding to the node index of
                    the nearest point on the grid.
        """
        return self._grid_dist2(p).argmin().item()

    def shortest_path(self, p1, p2):
        """Compute the shortest path on the discretized manifold.

        Inputs:
            p1:     a torch Tensor corresponding to one latent point.

            p2:     a torch Tensor corresponding to another latent point.

        Outputs:
            curve:  a DiscreteCurve forming the shortest path from p1 to p2.

            dist:   a scalar indicating the length of the shortest curve.
        """
        idx1 = self._grid_point(p1)
        idx2 = self._grid_point(p2)
        path = nx.shortest_path(self.G, source=idx1, target=idx2, weight='weight')  # list with N elements
        # coordinates = self.grid.view(self.grid.shape[0], -1)[:, path] # (dim)xN
        mesh = torch.meshgrid(*self.grid, indexing='ij')
        raw_coordinates = [m.flatten()[path].view(1, -1) for m in mesh]
        coordinates = torch.cat(raw_coordinates, dim=0)  # (dim)xN
        N = len(path)
        curve = DiscreteCurve(begin=coordinates[:, 0], end=coordinates[:, -1], num_nodes=N)
        with torch.no_grad():
            curve.parameters[:, :] = coordinates[:, 1:-1].t()
        dist = 0
        for i in range(N - 1):
            dist += self.G.edges[path[i], path[i + 1]]['weight']
        return curve, dist

    def connecting_geodesic(self, p1, p2, curve=None):
        """Compute the shortest path on the discretized manifold and fit
        a smooth curve to the resulting discrete curve.

        Inputs:
            p1:     a torch Tensor corresponding to one latent point.

            p2:     a torch Tensor corresponding to another latent point.

        Optional input:
            curve:  a curve that should be fitted to the discrete graph
                    geodesic. By default this is None and a CubicSpline
                    with default paramaters will be constructed.

        Outputs:
            curve:  a smooth curve forming the shortest path from p1 to p2.
                    By default the curve is a CubicSpline with its default
                    parameters; this can be changed through the optional
                    curve input.
        """
        device = p1.device
        if p1.ndim == 1:
            p1 = p1.unsqueeze(0) # 1xD
        if p2.ndim == 1:
            p2 = p2.unsqueeze(0) # 1xD
        B = p1.shape[0]
        if p1.shape != p2.shape:
            raise NameError('shape mismatch')

        if curve is None:
            curve = CubicSpline(p1, p2)
        else:
            curve.begin = p1
            curve.end = p2

        for b in range(B):
            idx1 = self._grid_point(p1[b].unsqueeze(0))
            idx2 = self._grid_point(p2[b].unsqueeze(0))
            path = nx.shortest_path(self.G, source=idx1, target=idx2, weight='weight')  # list with N elements
            weights = [self.G.edges[path[k], path[k + 1]]['weight'] for k in range(len(path) - 1)]
            mesh = torch.meshgrid(*self.grid, indexing='ij')
            raw_coordinates = [m.flatten()[path[1:-1]].view(-1, 1) for m in mesh]
            coordinates = torch.cat(raw_coordinates, dim=1)  # Nx(dim)
            t = torch.tensor(weights[:-1], device=device).cumsum(dim=0) / sum(weights)

            curve[b].fit(t, coordinates)

        return curve, True
