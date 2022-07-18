"""Euclidean manifold."""

from manifolds.base import Manifold
import torch

class Euclidean(Manifold):
    """
    Euclidean Manifold class.
    """

    def __init__(self):
        super(Euclidean, self).__init__()
        self.name = 'Euclidean'

    def normalize(self, p):
        dim = p.size(-1)
        p.view(-1, dim).renorm_(2, 0, 1.)
        return p

    def sqdist(self, p1, p2, c):
        return (p1 - p2).pow(2).sum(dim=-1)

    # ## 2020.08.16 temporalily try squared-lorentz distance
    # def sqdist(self, p1, p2, c):
    #     # print ("c=", c)
    #     x,y=p1,p2
    #     x2_srt = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + c)
    #     y2_srt = -torch.sqrt(torch.sum(y * y, dim=-1, keepdim=True) + c)
    #     u = torch.cat((x, x2_srt), -1)  # (N1, d+1)
    #     v = torch.cat((y, y2_srt), -1)  # (N2, d+1)
    #     assert u.shape == v.shape  # u,v should have same shape
    #     # uv = torch.sum(u * v, dim=-1, keepdim=True)  # (N1,1)
    #     uv = torch.sum(u * v, dim=-1)  # (N1)
    #     result = - 2 * c - 2 * uv
    #     return result

    def egrad2rgrad(self, p, dp, c):
        return dp

    def proj(self, p, c):
        return p

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        return p + u

    def logmap(self, p1, p2, c):
        return p2 - p1

    def expmap0(self, u, c):
        return u

    def logmap0(self, p, c):
        return p

    def mobius_add(self, x, y, c, dim=-1):
        return x + y

    def mobius_matvec(self, m, x, c):
        mx = x @ m.transpose(-1, -2)
        return mx

    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def inner(self, p, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, v, c):
        return v

    def ptransp0(self, x, v, c):
        return x + v
