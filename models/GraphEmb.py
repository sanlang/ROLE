
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

class GraphEmb(nn.Module):
    '''
        use the embedding method to predict next affected users, Inf2vec
    '''

    def __init__(self, opt,
                 dropout=0.1,
                 init_size=1e-3,
                 data_type="double",
                 score_type = "dot"
                 ):
        # super(CasBase_Emb, self).__init__(opt,num_embedding,dropout,init_size,data_type)
        super(GraphEmb, self).__init__()

        if data_type == 'double':
            self.data_type = torch.double
        else:
            self.data_type = torch.float
        self.init_size = init_size
        self.n_nodes = opt.n_nodes
        self.embedding_dim = opt.rank
        self.num_embedding = opt.num_embedding
        self.score_type = score_type
        self.num_negs = opt.neg_sample_size

        self.bias = opt.bias
        self.gamma = nn.Parameter(torch.Tensor([opt.gamma]), requires_grad=False)

        print ("Init GraphEmb....n_node= {}  dim={}  n_embedding={}".format(self.n_nodes, self.embedding_dim, self.num_embedding))

        # the last element is [vocab_size], which is used as padding_idx
        if self.num_embedding == 1:
            self.embed_s = nn.Embedding(self.n_nodes, self.embedding_dim)
            self.embed_s.weight.data = self.init_size * torch.randn((self.n_nodes, self.embedding_dim), dtype=self.data_type)
        elif self.num_embedding == 2:
            self.embed_s = nn.Embedding(self.n_nodes, self.embedding_dim)
            self.embed_t = nn.Embedding(self.n_nodes, self.embedding_dim)
            self.embed_s.weight.data = self.init_size * torch.randn((self.n_nodes, self.embedding_dim), dtype=self.data_type)
            self.embed_t.weight.data = self.init_size * torch.randn((self.n_nodes, self.embedding_dim), dtype=self.data_type)
        else:
            raise ValueError(
                "ValueError num_embedding: {0}".format(self.num_embedding)
            )

        if self.bias =="learn":
            self.bias_fr = nn.Embedding(self.n_nodes, 1)
            self.bias_fr.weight.data = torch.zeros((self.n_nodes, 1), dtype=self.data_type)
            self.bias_to = nn.Embedding(self.n_nodes, 1)
            self.bias_to.weight.data = torch.zeros((self.n_nodes, 1), dtype=self.data_type)

        if self.bias =="learn_fr":
            self.bias_fr = nn.Embedding(self.n_nodes, 1)
            self.bias_fr.weight.data = torch.zeros((self.n_nodes, 1), dtype=self.data_type)
        if self.bias == "learn_to":
            self.bias_to = nn.Embedding(self.n_nodes, 1)
            self.bias_to.weight.data = torch.zeros((self.n_nodes, 1), dtype=self.data_type)

    def read_embs(self, opt):
        assert opt.pre_train== True
        assert opt.pre_train_file is not None
        assert self.num_embedding==1
        print ("read_embes: ", opt.pre_train_file)
        node_embs={}
        dim=0
        with open (opt.pre_train_file, 'r') as f:
            lines = f.readlines()
            print (lines[0])
            node_num, dim = int(lines[0].strip().split(' ')[0]), int(lines[0].strip().split(' ')[1])
            for line in lines[1:]:
                strs = line.strip().split(' ')
                key = int(strs[0])
                vec = [float(i) for i in strs[1:]]
                node_embs[key] = vec
        print ("len(node_embs)=", len(node_embs))
        print ("self.n_nodes, self.embedding_dim: ", self.n_nodes, self.embedding_dim)
        print ("self.embed_s.weight.data.size(): ", self.embed_s.weight.data.size())
        assert self.embedding_dim==dim
        Non_exist_cnt=0
        for i in range(self.n_nodes):
            if i in node_embs:
                self.embed_s.weight.data[i,:] = torch.FloatTensor(node_embs[i])
            else:
                Non_exist_cnt +=1
                # print ("Not exist in embs:", i)
        print ("Non_exist_cnt=", Non_exist_cnt)

    def score(self, frs, tos):
        context_vecs = self.get_fr_embedding(frs) # b*l
        target_gold_vecs = self.get_to_embedding(tos) # b*t
        dist_score = self.similarity_score(context_vecs, target_gold_vecs)  # b*l*t
        if self.bias == 'constant':
            score = self.gamma.item() + dist_score
        elif self.bias =='learn':
            bias_frs = self.bias_fr(frs) # b*l*1
            bias_tos = self.bias_to(tos).permute(0,2,1) # b*t*1 ->b*1*t
            score =dist_score + bias_frs + bias_tos
        elif self.bias == 'learn_fr':
            bias_frs = self.bias_fr(frs)  # b*l*1
            # bias_tos = self.bias_to(tos).permute(0, 2, 1)  # b*t*1 ->b*1*t
            score = dist_score + bias_frs
        elif self.bias == 'learn_to':
            # bias_frs = self.bias_fr(frs)  # b*l*1
            bias_tos = self.bias_to(tos).permute(0, 2, 1)  # b*t*1 ->b*1*t
            score = dist_score + bias_tos

        else:
            # raise ValueError(
            #     "ValueError bias: {}".format(self.bias)
            # )
            score = dist_score

        return score
    
    def score_eval(self, frs,tos):
        return self.score(frs,tos)



    def similarity_score(self, context_vecs, target_vecs):
        '''
        # (b*L*d, b*t*d )-> b*L*t
        Args:
            context_vecs:
            target_vecs:
        Returns:
        '''
        if self.score_type == 'dot':  # (b*L*d, b*t*d )-> b*L*t
            target_vecs = target_vecs.permute(0, 2, 1)  # b*t*d -> b*d*t
            scores = torch.bmm(context_vecs, target_vecs)
            return scores
        elif self.score_type == 'eucl_dist':  # (b*L*d, b*t*d) -> b*L*t
            context_vecs = context_vecs.contiguous()
            target_vecs = target_vecs.contiguous()
            scores = torch.cdist(context_vecs, target_vecs, p=2)
            scores = scores.neg()
            return scores
        raise ValueError(
            "ValueError self.score_type: {0}".format(self.score_type)
        )


    def get_neg_samples(self, golds):
        '''
        sample negative examples
        Args:
            golds: b*1

        Returns:
        '''
        golds_size = golds.shape[0]
        tmp = golds.dtype
        assert golds.shape[1] == 1
        # neg_samples = torch.Tensor(np.random.randint(self.user_size,size=(int(golds_size), self.num_negs))
        # ).to(golds.dtype).cuda()
        neg_samples = torch.randint(self.n_nodes, (golds_size, self.num_negs)).cuda()
        # neg_samples = torch.arange(self.user_size-1).cuda()
        # neg_samples = torch.ones(16, 5).to(golds.dtype)

        return neg_samples

    def get_fr_embedding(self, frs):
        context_vecs = self.embed_s(frs)  # b*1*d
        return context_vecs
    def get_to_embedding(self, tos):
        if self.num_embedding == 1:
            target_vecs = self.embed_s(tos)  # b*1*d or b*s*d
        elif self.num_embedding == 2:
            target_vecs = self.embed_t(tos)  # b*1*d or b*s*d
        else:
            raise ValueError(
                "ValueError num_embedding: {0}".format(self.num_embedding)
            )
        return target_vecs
    
    



# use the inf_examples to train models
    def forward (self, input_batch):
        frs = input_batch[:,0:1]
        tos = input_batch[:,1:2]
        # eq= (frs==tos).sum()
        # print ("eq=", eq.item())


        # batch_size = input_batch.size(0)
        # batch_loss = 0.0

        # frs = torch.unsqueeze(frs, 1)
        # context_vecs = self.embed_s(frs) # b*1*d
        # target_gold = torch.unsqueeze(tos, 1)  # b*1
        # target_neg = self.get_neg_samples(target_gold)  # negative samples b*n
        #
        # if self.num_embedding == 1:
        #     target_gold_vecs = self.embed_s(target_gold)  # b*1*d
        #     target_neg_vecs = self.embed_s(target_neg)  # b*neg*d
        # elif self.num_embedding == 2:
        #     target_gold_vecs = self.embed_t(target_gold)  # b*1*d
        #     target_neg_vecs = self.embed_t(target_neg)  # b*neg*d
        # else:
        #     raise ValueError(
        #         "ValueError num_embedding: {0}".format(self.num_embedding)
        #     )

        # context_vecs = self.get_fr_embedding(frs)
        # target_gold_vecs = self.get_to_embedding(tos)
        # target_neg = self.get_neg_samples(tos)  # negative samples b*n
        # target_neg_vecs = self.get_to_embedding(target_neg)
        # positive_score = self.similarity_score(context_vecs, target_gold_vecs)  # b*1*1
        # negative_score = self.similarity_score(context_vecs, target_neg_vecs)  # b*1*neg
        # # print ("\n torch.isnan(positive_score).int().sum()= ", torch.isnan(positive_score).int().sum().item())
        # # print("\n torch.isnan(negative_score).int().sum()= ", torch.isnan(negative_score).int().sum().item())

        to_negs = self.get_neg_samples(tos)
        positive_score = self.score(frs,tos)
        negative_score = self.score(frs, to_negs)

        positive_loss = F.logsigmoid(positive_score).sum()  # b*L*1 -> num
        negative_loss = F.logsigmoid(-negative_score).sum()  # b*L*neg -> num

        batch_loss = - (positive_loss + negative_loss)
        # print (batch_loss.item(), positive_loss.item(), negative_loss.item())

        # positive_loss = F.logsigmoid(positive_score)  # b*L*1 -> num
        # negative_loss = F.logsigmoid(-negative_score)  # b*L*neg -> num
        # loss = - torch.cat([positive_score, negative_score], dim=-1).mean()
        # batch_loss = loss
        return batch_loss


    def get_edge_score(self, test_edges):
        assert test_edges.size(1)==2
        frs = test_edges[:, 0:1]
        tos = test_edges[:, 1:2]
        # frs = torch.unsqueeze(frs, 1)
        # context_vecs = self.embed_s(frs)  # b*1*d
        # target_gold = torch.unsqueeze(tos, 1)  # b*1

        # context_vecs = self.get_fr_embedding(frs)
        # target_gold_vecs = self.get_to_embedding(tos)
        # edge_score = self.similarity_score(context_vecs, target_gold_vecs)  # b*1*1
        edge_score = self.score_eval(frs,tos)
        return edge_score

    def get_noderec_score(self, noderec_batch):
        '''
        For each edge, evaluate ground-truth against sampled negative nodes
        Args:
            noderec_batch:

        Returns:
        '''
        frs = noderec_batch[:,:1] # (b, 1)
        tos = noderec_batch[:,1:] # (b, s)
        # context_vecs = self.embed_s(frs)  # b*1*d
        # if self.num_embedding == 1:
        #     target_gold_vecs = self.embed_s(tos)  # b*s*d
        # elif self.num_embedding == 2:
        #     target_gold_vecs = self.embed_t(tos)  # b*s*d
        # else:
        #     raise ValueError(
        #         "ValueError num_embedding: {0}".format(self.num_embedding)
        #     )

        # context_vecs = self.get_fr_embedding(frs)
        # target_gold_vecs = self.get_to_embedding(tos)
        # noderec_score = self.similarity_score(context_vecs, target_gold_vecs)  # b*1*s
        noderec_score = self.score_eval(frs,tos)

        noderec_score = noderec_score.squeeze()
        return noderec_score


    def get_embeddings(self):
        x= self.embed_s.weight
        return x

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])


class Dot (GraphEmb):
    '''
    use the Dot Product, can be used for APP and node2vec
    '''
    def __init__(self, args):
        super(Dot, self).__init__(args)
        self.score_type = "dot"


class Node2vec (GraphEmb):
    '''
    use the Dot Product, can be used for APP and node2vec
    '''
    def __init__(self, args):
        super(Node2vec, self).__init__(args)
        self.score_type = "dot"
        assert self.num_embedding ==2

    # ## Different from get_fr_embedding and get_to_embedding, this is used for evaluation
    # def get_fr_embedding_eval(self, frs):
    #     context_vecs = self.get_fr_embedding(frs)  # b*1*d
    #     return context_vecs
    #
    # def get_to_embedding_eval(self, tos):
    #     target_vecs = self.get_to_embedding(tos)
    #     return target_vecs
    
    
    
    def score_eval(self, frs,tos):
        '''
        This is different from score(), to use the same setting of node2vec
        '''
        # context_vecs = self.get_fr_embedding(frs)  # b*l
        # target_gold_vecs = self.get_to_embedding(tos)  # b*t
        # context_vecs = self.embed_s(frs)
        context_vecs = self.embed_s(frs)  # b*l
        target_gold_vecs = self.embed_s(tos)
        dist_score = self.similarity_score(context_vecs, target_gold_vecs)  # b*l*t
        if self.bias == 'constant':
            score = self.gamma.item() + dist_score
        elif self.bias == 'learn':
            bias_frs = self.bias_fr(frs)  # b*l*1
            bias_tos = self.bias_to(tos).permute(0, 2, 1)  # b*t*1 ->b*1*t
            score = dist_score + bias_frs + bias_tos
        elif self.bias == 'learn_fr':
            bias_frs = self.bias_fr(frs)  # b*l*1
            # bias_tos = self.bias_to(tos).permute(0, 2, 1)  # b*t*1 ->b*1*t
            score = dist_score + bias_frs
        elif self.bias == 'learn_to':
            # bias_frs = self.bias_fr(frs)  # b*l*1
            bias_tos = self.bias_to(tos).permute(0, 2, 1)  # b*t*1 ->b*1*t
            score = dist_score + bias_tos
    
        else:
            # raise ValueError(
            #     "ValueError bias: {}".format(self.bias)
            # )
            score = dist_score
    
        return score
    
        

class ME (GraphEmb):
    '''
    use the Euclidean distance
    '''
    def __init__(self, args):
        super(ME, self).__init__(args)
        self.score_type = "eucl_dist"

class Node2lv (GraphEmb):
    '''
    use the squared lorentz distance
    '''
    def __init__(self, args):
        super(Node2lv, self).__init__(args)
        self.score_type = "lorentz_dist"
        self.beta = args.beta

    def similarity_score(self, context_vecs, target_vecs):
        '''
        # (b*L*d, b*t*d )-> b*L*t
        Args:
            context_vecs:
            target_vecs:
        Returns:
        '''
        if self.score_type == 'lorentz_dist':
            c = self.beta
            x = context_vecs #b*L*d
            y = target_vecs # b*t*d
            x2_srt = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + c) # b*L*1
            y2_srt = -torch.sqrt(torch.sum(y * y, dim=-1, keepdim=True) + c) # b*t*1
            u = torch.cat((x, x2_srt), -1)  # (b,L, d+1)
            v = torch.cat((y, y2_srt), -1)  # (b,t, d+1)

            # u,v have different shape[0]
            vt = v.permute(0, 2, 1)  # (b, d+1,t)
            # scores = torch.bmm(context_vecs, target_vecs)
            uv = torch.bmm(u, vt)  # (b, L, t)
            # if train_model:
            #     # u,v have different shape[0]
            #     vt = v.permute(0, 2, 1) # (b, d+1,t)
            #     #scores = torch.bmm(context_vecs, target_vecs)
            #     uv = torch.bmm(u, vt) # (b, L, t)
            #
            # else:
            #     assert u.shape == v.shape  # u,v should have same shape, (b,L,d)
            #     uv = torch.sum(u * v, dim=-1, keepdim=True)  # (b,L,1)
            result = - 2 * c - 2 * uv
            score = result.neg()
            return score
        else:
            raise ValueError(
                "ValueError score_type: {0}".format(self.score_type)
            )


# ################# Hyperbolic MATH FUNCTIONS ########################
# copy from KGEmb-acl-20
# 2020.08.27 This code cannot work, "NaN, infinity or a value too large"
MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5, torch.double: 1e-6}
class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)

def tanh(x):
    return x.clamp(-15, 15).tanh()

def expmap0(u, c):
    """Exponential map taken at the origin of the Poincare ball with curvature c.

    Args:
        u: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with tangent points.
    """
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)

def project(x, c):
    """Project points to Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic points
        c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

    Returns:
        torch.Tensor with projected hyperbolic points.
    """
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)

def hyp_distance(x, y, c, eval_mode=False):
    """Hyperbolic distance on the Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic queries
        y: torch.Tensor with hyperbolic queries, shape n_entities x d if eval_mode is true else (B x d)
        c: torch.Tensor of size 1 with absolute hyperbolic curvature

    Returns: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    sqrt_c = c ** 0.5
    x2 = torch.sum(x * x, dim=-1, keepdim=True) # B*1
    if eval_mode:
        y2 = torch.sum(y * y, dim=-1, keepdim=True).transpose(0, 1) # 1*n_entities
        xy = x @ y.transpose(0, 1) # B*d, d*n_entities -> B*n_entities
    else:
        y2 = torch.sum(y * y, dim=-1, keepdim=True) # B*1
        xy = torch.sum(x * y, dim=-1, keepdim=True) # B*d, B*d ->B*1
    c1 = 1 - 2 * c * xy + c * y2 #
    c2 = 1 - c * x2
    num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * y2 - (2 * c1 * c2) * xy) # vics: this is correct, sqrt(x^2) -> ||x||
    denom = 1 - 2 * c * xy + c ** 2 * x2 * y2 # vics: since this term is always positive, denom>0
    pairwise_norm = num / denom.clamp_min(MIN_NORM)
    dist = artanh(sqrt_c * pairwise_norm)
    return 2 * dist / sqrt_c

def hyp_distance_own(x,y,c):
    '''
    2020.09.21 My own way to implement hyp_distance. Note that x,y are poincare ball embedding
    Args:
        x: b*L*d
        y:  b*t*d
        c: 1

    Returns:
        distance: b*L*t
    '''
    sqrt_c = c ** 0.5
    x2 = torch.sum(x * x, dim=-1, keepdim=True)  # b*L*1
    y2 = torch.sum(y * y, dim=-1, keepdim=True).permute(0, 2, 1)  # b*t*d ->b*t*1 ->b*1*t
    y_permute = y.permute(0, 2, 1)  # b*d*t
    xy = torch.bmm(x, y_permute)  # (b,L,d)*(b,d,t) -> b * L *t

    c1 = 1 - 2 * c * xy + c * y2  # b*L*t
    c2 = 1 - c * x2  # b*L*1

    # num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * y2 - (2 * c1 * c2) * xy)  # (b,L,t)
    num2 = (c1 ** 2) * x2 + (c2 ** 2) * y2 - (2 * c1 * c2) * xy
    num = torch.sqrt(num2.clamp_min(MIN_NORM))  # # make sure that x>=0
    denom = 1 - 2 * c * xy + (c ** 2) * x2 * y2  # x2 * y2: (b*L*1) * (b*1*t) => (b*L*t)
    pairwise_norm = num / denom.clamp_min(MIN_NORM)
    dist = artanh(sqrt_c * pairwise_norm)
    result = 2 * dist / sqrt_c
    return  result




class Poincare_acl20(GraphEmb):
    '''
    use the Poincare distance, using tangent space operation, norm optimizer
    '''
    def __init__(self, args):
        super(Poincare_acl20, self).__init__(args)
        self.score_type = "poincare_dist"
        self.beta = args.beta
        assert args.optimizer == "Adam"
        # self.c = 1.0/args.beta


    def similarity_score(self, context_vecs, target_vecs):
        '''
        # (b*L*d, b*t*d )-> b*L*t
        Args:
            context_vecs:
            target_vecs:
        Returns:
        '''
        if self.score_type == 'poincare_dist':
            c= 1.0/self.beta
            assert context_vecs.size(1) == 1 # the second dim is always 1, (b,1,d)


            # Implementation (1)
            # the following code is copied from KGEmb, use Euclidean optimizer Adam
            # Note that context_vecs/target_vecs should be converted to poincare space
            Dim = context_vecs.size(-1)
            cvs = context_vecs.view(-1, Dim)
            tvs = target_vecs.view(-1, Dim)
            x = expmap0(cvs, c).view(context_vecs.size())
            y = expmap0(tvs, c).view(target_vecs.size())

            # ## Implementation (2)
            # ## directly use poincare space, should use RiemannianAdam
            # x = context_vecs
            # y = target_vecs

            ## the way to implement poincare_distance
            # sqrt_c = c ** 0.5
            # x2 = torch.sum(x*x, dim=-1, keepdim=True) # b*L*1
            # y2 = torch.sum(y*y, dim=-1, keepdim=True).permute(0,2,1) # b*t*d ->b*t*1 ->b*1*t
            # y_permute = y.permute(0, 2, 1) # b*d*t
            # xy = torch.bmm(x, y_permute) # (b,L,d)*(b,d,t) -> b * L *t
            #
            # c1 = 1 - 2 * c * xy + c * y2  # b*L*t
            # c2 = 1 - c * x2 # b*L*1
            #
            # # num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * y2 - (2 * c1 * c2) * xy)  # (b,L,t)
            # num2= (c1 ** 2) * x2 + (c2 ** 2) * y2 - (2 * c1 * c2) * xy
            # num = torch.sqrt(num2.clamp_min(MIN_NORM)) # # make sure that x>=0
            # denom = 1 - 2 * c * xy + (c ** 2) * x2 * y2 #  x2 * y2: (b*L*1) * (b*1*t) => (b*L*t)
            # pairwise_norm = num / denom.clamp_min(MIN_NORM)
            # dist = artanh(sqrt_c * pairwise_norm)
            # result = 2 * dist / sqrt_c

            result = hyp_distance_own(x,y,c)

            return  (result**2).neg() # (b*L*t)

            # # Test: Use the function
            # x= x.squeeze()
            # y= y[:,:1,:].squeeze()
            # dist= hyp_distance(x,y,c)
            # return dist.neg()

        else:
            raise ValueError(
                "ValueError score_type: {0}".format(self.score_type)
            )
    
    
        
            
        



class Poincare_nips19_radam(GraphEmb):
    '''
    use the radam from NIPS-19 (hgcn)
    '''
    def __init__(self, args):
        super( Poincare_nips19_radam, self).__init__(args)
        self.score_type = "poincare_dist"
        self.beta = args.beta
        # self.c = 1.0/args.beta
        assert args.optimizer == "RiemannianAdam"

    def similarity_score(self, context_vecs, target_vecs):
        '''
        # (b*L*d, b*t*d )-> b*L*t
        Args:
            context_vecs:
            target_vecs:
        Returns:
        '''
        if self.score_type == 'poincare_dist':
            c= 1.0/self.beta
            assert context_vecs.size(1) == 1 # the second dim is always 1, (b,1,d)
            sqrt_c = c ** 0.5

            # Implementation (1)
            # the following code is copied from KGEmb, use Euclidean optimizer Adam
            # Note that context_vecs/target_vecs should be converted to poincare space
            Dim = context_vecs.size(-1)
            cvs = context_vecs.view(-1, Dim)
            tvs = target_vecs.view(-1, Dim)
            x = expmap0(cvs, c).view(context_vecs.size())
            y = expmap0(tvs, c).view(target_vecs.size())

            ## Implementation (2)
            ## directly use poincare space, should use RiemannianAdam
            x = context_vecs
            y = target_vecs

            x2 = torch.sum(x*x, dim=-1, keepdim=True) # b*L*1
            y2 = torch.sum(y*y, dim=-1, keepdim=True).permute(0,2,1) # b*t*d ->b*t*1 ->b*1*t
            y_permute = y.permute(0, 2, 1) # b*d*t
            xy = torch.bmm(x, y_permute) # (b,L,d)*(b,d,t) -> b * L *t

            c1 = 1 - 2 * c * xy + c * y2  # b*L*t
            c2 = 1 - c * x2 # b*L*1

            # num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * y2 - (2 * c1 * c2) * xy)  # (b,L,t)
            num2= (c1 ** 2) * x2 + (c2 ** 2) * y2 - (2 * c1 * c2) * xy
            num = torch.sqrt(num2.clamp_min(MIN_NORM)) # # make sure that x>=0
            denom = 1 - 2 * c * xy + (c ** 2) * x2 * y2 #  x2 * y2: (b*L*1) * (b*1*t) => (b*L*t)
            pairwise_norm = num / denom.clamp_min(MIN_NORM)
            dist = artanh(sqrt_c * pairwise_norm)
            result = 2 * dist / sqrt_c

            return  (result**2).neg() # (b*L*t)

            # # Test: Use the function
            # x= x.squeeze()
            # y= y[:,:1,:].squeeze()
            # dist= hyp_distance(x,y,c)
            # return dist.neg()

        else:
            raise ValueError(
                "ValueError score_type: {0}".format(self.score_type)
            )






