
'''2020.08.30 Improve the embedding for directed_graph
'''
import torch.nn as nn
import torch

from .GraphEmb import GraphEmb, Node2lv

def givens_rotations(r, x):
    """Givens rotations.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate

    Returns:
        torch.Tensor os shape (N x d) representing rotation of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True)
    x = x.view((r.shape[0], -1, 2))
    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1) # the first: cos, second: sin
    return x_rot.view((r.shape[0], -1))


class ROLE(Node2lv):
    def __init__(self, opt):
        super(ROLE, self).__init__(opt)
        assert opt.num_embedding == 1
        assert opt.directed == True
        self.rel_diag = nn.Embedding(2, self.embedding_dim)
        self.rel_diag.weight.data = 2 * torch.rand((2, self.embedding_dim), dtype=self.data_type) - 1.0

        # set the rotation to be degree 0. [1,0]
        self.rel_diag.weight.data[:,::2]=1.0
        self.rel_diag.weight.data[:, 1::2] = 0.0

    def get_fr_embedding(self, frs):
        context_vecs = self.embed_s(frs)  # b*1*d
        dim = context_vecs.size(2)

        # rel_diag_vecs =self.fr_node_rel_diag(context_frs) # b*1*d
        context_frs = torch.zeros(frs.size(), dtype=torch.long).cuda()  # b*1
        rel_diag_vecs = self.rel_diag(context_frs)
        context_rot = givens_rotations(rel_diag_vecs.view(-1,dim), context_vecs.view(-1,dim)).view(context_vecs.size())
        return context_rot

    def get_to_embedding(self, tos):
        context_vecs = self.embed_s(tos)  # b*s*d
        dim = context_vecs.size(2)

        # rel_diag_vecs =self.to_node_rel_diag(tos) # b*s*d
        context_tos = torch.ones(tos.size(), dtype=torch.long).cuda()  # b*s
        rel_diag_vecs = self.rel_diag(context_tos)  # b*s*d
        context_rot = givens_rotations(rel_diag_vecs.view(-1,dim), context_vecs.view(-1,dim)).view(context_vecs.size())
        return context_rot

