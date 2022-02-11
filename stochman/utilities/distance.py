#!/usr/bin/env python3
import torch


class __Dist2__(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, p0: torch.Tensor, p1: torch.Tensor):
        with torch.no_grad():
            with torch.enable_grad():
                # TODO: Only perform the computations needed for backpropagation
                # (check if p0.requires_grad and p1.requires_grad)
                C, success = M.connecting_geodesic(p0, p1)

                lm0 = C.deriv(torch.zeros(1, device=p0.device)).squeeze(1)  # log(p0, p1); Bx(d)
                lm1 = -C.deriv(torch.ones(1, device=p0.device)).squeeze(1)   # log(p1, p0); Bx(d)
                M0 = M.metric(p0)  # Bx(d)x(d) or Bx(d)
                M1 = M.metric(p1)  # Bx(d)x(d) or Bx(d)
                if M0.ndim == 3:  # metric is square
                    Mlm0 = M0.bmm(lm0.unsqueeze(-1)).squeeze(-1)  # Bx(d)
                    Mlm1 = M1.bmm(lm1.unsqueeze(-1)).squeeze(-1)  # Bx(d)
                else:
                    Mlm0 = M0 * lm0  # Bx(d)
                    Mlm1 = M1 * lm1  # Bx(d)

                ctx.save_for_backward(Mlm0, Mlm1)
                retval = (lm0 * Mlm0).sum(dim=-1)  # B
        return retval

    @staticmethod
    def backward(ctx, grad_output):
        Mlm0, Mlm1 = ctx.saved_tensors
        return (None,
                2.0 * grad_output.view(-1, 1) * Mlm0,
                2.0 * grad_output.view(-1, 1) * Mlm1)


def squared_manifold_distance(manifold, p0: torch.Tensor, p1: torch.Tensor):
    """
    Computes the squared distance between point p0 and p1 on the manifold

    Args:
        manifold: manifold object
        p0: initial point
        p1: end point
    Returns:
        dist: squared distance from p0 to p1 calculated on the manifold
    """
    distance_op = __Dist2__()
    return distance_op.apply(manifold, p0, p1)
