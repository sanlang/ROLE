
import torch.nn as nn
import torch

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

