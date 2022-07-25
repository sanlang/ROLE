'''2020.08.31 Implement Complex Embedding for directed graph'''
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

from .GraphEmb import GraphEmb

# class ComplexBase (GraphEmb):
#     '''
#     use the Euclidean distance
#     '''
#     def __init__(self, args):
#         super(ComplexBase, self).__init__(args)
#         self.score_type = "complex"
#         assert self.embedding_dim % 2 == 0, "Complex models require even embedding dimension"
#         self.rank = args.rank // 2
#         assert self.embedding_dim == 2* self.rank
#
#     def similarity_score(self, context_vecs, target_vecs):
#         '''
#         # (b*L*d, b*t*d )-> b*L*t
#         Args:
#             context_vecs:
#             target_vecs:
#         Returns:
#         '''
#         if self.score_type=='complex':
#             context_real = context_vecs[:,:,:self.rank] # real
#             context_imaginary = context_vecs[:,:,self.rank:] # imaginary
#
#             target_real = target_vecs[:,:,:self.rank]
#             target_imaginary = target_vecs[:,:, self.rank:]
#
#             # Same as complex: Re(<<h,r>,t_>), only consider the real part of
#             score = torch.bmm(context_real, target_real.permute(0,2,1)) + torch.bmm(context_imaginary, target_imaginary.permute(0,2,1)) # b*L*t
#             return score
#         raise ValueError(
#             "ValueError self.score_type: {0}".format(self.score_type)
#         )

class RotatE_Each(GraphEmb):
    '''
    Each node has a rel_embedding
    '''
    def __init__(self, args):
        super(RotatE_Each, self).__init__(args)
        self.score_type = "RotatE"
        assert self.num_embedding ==1
        assert self.embedding_dim % 2 == 0, "Complex models require even embedding dimension"
        self.rank = args.rank // 2
        assert self.embedding_dim == 2* self.rank
        self.rel_embedding = nn.Embedding(self.n_nodes, self.embedding_dim)
        self.rel_embedding.weight.data = self.init_size * torch.randn((self.n_nodes, self.embedding_dim),
                                                                dtype=self.data_type)

    def get_fr_embedding(self, frs):
        rel_vecs = self.rel_embedding(frs)
        rel_vecs_r = rel_vecs[:,:,:self.rank]
        rel_vecs_i = rel_vecs[:,:,self.rank:]
        rel_norm = torch.sqrt(rel_vecs_r** 2 + rel_vecs_i ** 2)
        cos = rel_vecs_r / rel_norm
        sin = rel_vecs_i / rel_norm

        context_vecs = self.embed_s(frs)  # b*1*d
        context_real = context_vecs[:, :, :self.rank]  # real
        context_imaginary = context_vecs[:, :, self.rank:]  # imaginary

        context_vecs_dot = torch.cat([
            context_real * cos - context_imaginary * sin,
            context_real * sin + context_imaginary * cos
        ], -1)
        return context_vecs_dot

    def similarity_score(self, context_vecs, target_vecs):
        '''
        # (b*L*d, b*t*d )-> b*L*t
        Args:
            context_vecs:
            target_vecs:
        Returns:
        '''
        if self.score_type=='RotatE':
            assert context_vecs.size(1) ==1 # should be b*1*d
            re_score = context_vecs[:,:,:self.rank] - target_vecs[:,:,:self.rank]
            im_score = context_vecs[:, :, self.rank:] - target_vecs[:, :,self.rank:]

            score = torch.stack([re_score,im_score], dim =0) # 2*b*s*d
            score = score.norm(dim=0)  # b*s*d
            dist = score.sum(dim=-1) # b*s
            result = -dist.unsqueeze(1) # b*1*s
            return result

            # score = torch.stack([re_score, im_score], dim=0)
            # score = score.norm(dim=0)
            # score = self.gamma.item() - score.sum(dim=2)

        raise ValueError(
            "ValueError self.score_type: {0}".format(self.score_type)
        )


class RotatE(GraphEmb):
    '''
    2020.09.21 Has only one relation vector
    '''
    def __init__(self, args):
        super(RotatE, self).__init__(args)
        self.score_type = "RotatE"
        assert self.num_embedding ==1
        assert self.embedding_dim % 2 == 0, "Complex models require even embedding dimension"
        self.rank = args.rank // 2
        assert self.embedding_dim == 2* self.rank
        # only one embedding to learn
        self.rel_embedding = nn.Embedding(1, self.embedding_dim)
        self.rel_embedding.weight.data = self.init_size * torch.randn((1, self.embedding_dim),
                                                                dtype=self.data_type)

    def get_fr_embedding(self, frs):
        rel_idx = torch.zeros(frs.size(), dtype=torch.long).cuda()
        rel_vecs = self.rel_embedding(rel_idx)

        rel_vecs_r = rel_vecs[:,:,:self.rank]
        rel_vecs_i = rel_vecs[:,:,self.rank:]
        rel_norm = torch.sqrt(rel_vecs_r** 2 + rel_vecs_i ** 2)
        cos = rel_vecs_r / rel_norm
        sin = rel_vecs_i / rel_norm

        context_vecs = self.embed_s(frs)  # b*1*d
        context_real = context_vecs[:, :, :self.rank]  # real
        context_imaginary = context_vecs[:, :, self.rank:]  # imaginary

        context_vecs_dot = torch.cat([
            context_real * cos - context_imaginary * sin,
            context_real * sin + context_imaginary * cos
        ], -1)
        return context_vecs_dot

    def similarity_score(self, context_vecs, target_vecs):
        '''
        # (b*L*d, b*t*d )-> b*L*t
        Args:
            context_vecs:
            target_vecs:
        Returns:
        '''
        if self.score_type=='RotatE':
            assert context_vecs.size(1) ==1 # should be b*1*d
            re_score = context_vecs[:,:,:self.rank] - target_vecs[:,:,:self.rank]
            im_score = context_vecs[:, :, self.rank:] - target_vecs[:, :,self.rank:]

            score = torch.stack([re_score,im_score], dim =0) # 2*b*s*d
            score = score.norm(dim=0)  # b*s*d
            dist = score.sum(dim=-1) # b*s
            result = -dist.unsqueeze(1) # b*1*s
            return result

            # score = torch.stack([re_score, im_score], dim=0)
            # score = score.norm(dim=0)
            # score = self.gamma.item() - score.sum(dim=2)

        raise ValueError(
            "ValueError self.score_type: {0}".format(self.score_type)
        )

class ModE(GraphEmb):
    ''' The part of HAKE, only consider modulus part'''
    def __init__(self, args):
        super(ModE, self).__init__(args)
        self.score_type = "ModE"
        assert self.num_embedding == 1
        self.rel_embedding = nn.Embedding(self.n_nodes, self.embedding_dim)
        self.rel_embedding.weight.data = self.init_size * torch.randn((self.n_nodes, self.embedding_dim),
                                                                      dtype=self.data_type)

    def get_fr_embedding(self, frs):
        rel_vecs = self.rel_embedding(frs)
        context_vecs =self.embed_s(frs)
        return rel_vecs*context_vecs

    def similarity_score(self, context_vecs, target_vecs):
        '''
        # (b*L*d, b*t*d )-> b*L*t
        Args:
            context_vecs:
            target_vecs:
        Returns:
        '''
        if self.score_type=='ModE':
            assert context_vecs.size(1) ==1 # should be b*1*d
            diff = context_vecs - target_vecs # b*t*d
            dist = torch.norm(diff, p=1, dim=2) # b*t
            result = -dist.unsqueeze(1) # b*1*t
            return result

        raise ValueError(
            "ValueError self.score_type: {0}".format(self.score_type)
        )

class TransE(GraphEmb):
    ''' The part of HAKE, only consider modulus part'''
    def __init__(self, args):
        super(TransE, self).__init__(args)
        self.score_type = "TransE"
        assert self.num_embedding == 1
        self.rel_embedding = nn.Embedding(self.n_nodes, self.embedding_dim)
        self.rel_embedding.weight.data = self.init_size * torch.randn((self.n_nodes, self.embedding_dim),
                                                                      dtype=self.data_type)

    def get_fr_embedding(self, frs):
        rel_vecs = self.rel_embedding(frs)
        context_vecs =self.embed_s(frs)
        return rel_vecs+context_vecs

    def similarity_score(self, context_vecs, target_vecs):
        '''
        # (b*L*d, b*t*d )-> b*L*t
        Args:
            context_vecs:
            target_vecs:
        Returns:
        '''
        if self.score_type=='TransE':
            assert context_vecs.size(1) ==1 # should be b*1*d
            diff = context_vecs - target_vecs # b*t*d
            dist = torch.norm(diff, p=1, dim=2) # b*t
            result = -dist.unsqueeze(1) # b*1*t
            return result

        raise ValueError(
            "ValueError self.score_type: {0}".format(self.score_type)
        )


class HAKE(GraphEmb):
    '''
    Each node has a rel_embedding
    '''
    def __init__(self, args):
        super(HAKE, self).__init__(args)
        self.score_type = "HAKE"
        assert self.embedding_dim % 2 == 0, "Complex models require even embedding dimension"
        self.rank = args.rank // 2
        assert self.embedding_dim == 2* self.rank
        self.rel_embedding = nn.Embedding(self.n_nodes, 3* self.rank) # phase, mod and bias
        self.rel_embedding.weight.data = self.init_size * torch.randn((self.n_nodes, 3* self.rank),
                                                                dtype=self.data_type)
        # nn.init.ones_(
        #     tensor=self.rel_embedding.weight.data[:, self.rank:2 * self.rank]
        # )
        #
        # nn.init.zeros_(
        #     tensor=self.rel_embedding.weight.data[:, 2 * self.rank: 3*self.rank]
        # )
        self.pi = 3.14159262358979323846
        self.phase_weight =0.5
        self.modulus_weight =0.5
        self.embedding_range = torch.tensor([self.init_size])

    def get_fr_embedding(self, frs):
        context_vecs = self.embed_s(frs)
        rel_vecs = self.rel_embedding(frs)

        return context_vecs, rel_vecs

    def similarity_score(self, context_vecs, target_vecs):
        '''
        context_vecs contains: context_vecs, rel_vecs
        '''
        if self.score_type=='HAKE':
            head = context_vecs[0]
            relation = context_vecs[1]
            tail = target_vecs # b*t*d
            head = head.expand_as(tail) # b*1*d -> b*t*d
            relation_size=[relation.size(0),tail.size(1), relation.size(2) ] # b*t*(1.5*d)
            rel = relation.expand(relation_size)

            phase_head, mod_head = torch.chunk(head, 2, dim=2)
            phase_relation, mod_relation, bias_relation = torch.chunk(rel, 3, dim=2)
            phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)

            phase_head = phase_head / (self.embedding_range.item() / self.pi)
            phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
            phase_tail = phase_tail / (self.embedding_range.item() / self.pi)

            phase_score = (phase_head + phase_relation) - phase_tail
            mod_relation = torch.abs(mod_relation)
            bias_relation = torch.clamp(bias_relation, max=1)
            indicator = (bias_relation < -mod_relation)
            bias_relation[indicator] = -mod_relation[indicator]

            r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

            phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
            r_score = torch.norm(r_score, dim=2) * self.modulus_weight
            return - (phase_score + r_score)

        raise ValueError(
            "ValueError self.score_type: {0}".format(self.score_type)
        )






