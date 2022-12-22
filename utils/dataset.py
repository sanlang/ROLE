import networkx as nx
import numpy as np
import scipy.sparse as sp
import sys
import random
import torch

import os

def read_graph(args, input_file):
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(input_file, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(input_file, nodetype=int, create_using=nx.DiGraph())
        # for edge in G.edges():
        # 	G[edge[0]][edge[1]]['weight'] = 1
    if not args.directed:
        G = G.to_undirected()

    print(input_file)
    print (G.number_of_nodes(), G.number_of_edges())
    return G

def load_graph_data(args, dataset_str):
    """ Load social network as a sparse adjacency matrix. """
    print("Loading graph", dataset_str)

    graph_data = {}
    for split in ["all", "train", "test", "valid"]:
        file_path = os.path.join(dataset_str, args.dataset+"_"+split + ".txt")
        graph_data[split] = read_graph(args,file_path)

    all_edges = list(graph_data["all"].edges)
    train_edges = list(graph_data["train"].edges)
    valid_edges = list(graph_data["valid"].edges)
    test_edges = list(graph_data["test"].edges)

    return graph_data, torch.LongTensor(all_edges), torch.LongTensor(train_edges), torch.LongTensor(valid_edges),torch.LongTensor(test_edges)

# 2020.09.08 Read the heterogenous graph
def load_hetero_graph_data(args, dataset_str):
    """ Load social network as a sparse adjacency matrix. """
    print("load_hetero_graph_data", dataset_str)

    # Read node-node relations
    graph_data = {}
    for split in ["all", "train", "test", "valid"]:
        file_path = os.path.join(dataset_str, args.dataset+"_"+split + ".txt")
        graph_data[split] = read_graph(args,file_path)

    all_edges = list(graph_data["all"].edges)
    train_edges = list(graph_data["train"].edges)
    valid_edges = list(graph_data["valid"].edges)
    test_edges = list(graph_data["test"].edges)

    # Read node-type
    node_type_path = os.path.join(dataset_str, args.dataset + "_" + "node_type.txt")
    dict_node_type = {}
    set_cat =set()
    with open(node_type_path, 'r') as inputfile:
        for line in inputfile:
            strs = line.strip().split()
            node, ty = int(strs[0]), strs[1]
            dict_node_type[node]= ty
            if ty!='p':
                set_cat.add(node)
    print ("len(dict_node_type)=", len(dict_node_type))
    print ("len(set_cat)=",len(set_cat))

    dict_cat_idx={} # resign id from 0
    catidx_type={} # new category id with its type
    idx=0
    for cat in set_cat:
        dict_cat_idx[cat]=idx
        catidx_type[idx] = dict_node_type[cat]
        idx+=1
    
    ## Temporal: extract the oldcatid_newcatid
    # with open("tree_directed_cora_oldcatid_newcatid.txt",'w') as fout:
    #     for k,v in dict_cat_idx.items():
    #         fout.write(str(k)+"\t"+str(v)+"\n")



    ## read the node-cat relation
    node_catnodes_path = os.path.join(dataset_str, args.dataset + "_" + "nodeid_catnodes.txt")
    dict_node_cats={}
    with open(node_catnodes_path, 'r') as inputfile:
        for line in inputfile:
            strs = line.strip().split(":")
            node, cats = int(strs[0]), strs[1]
            cats_idx =[dict_cat_idx[int(c)] for c in cats.split(" ")] # all category
            # cat_list = cats.split(" ")
            # cats_idx =[dict_cat_idx[int(cat_list[-1])] ] # last category
            # cat_list = cats.split(" ")
            # cats_idx =[dict_cat_idx[int(cat_list[0])] ] # first category
            dict_node_cats[node] = cats_idx
    print("len(dict_node_cats)=", len(dict_node_cats))

    train_edges_pc =[]
    train_edges_cc=[]
    tree_train_edges_cc=set()
    train_edges_cc_set =set()
    train_nodes=list(graph_data["train"].nodes) # only consider the nodes in train dataset
    for p in train_nodes:
        for c in dict_node_cats[p]:
            train_edges_pc.append([p,c])

        # get the c-c relations
        cats = dict_node_cats[p]
        cat_len=len(cats)

        if cat_len>1:
            for i in range(cat_len-1):
                tree_train_edges_cc.add((cats[i],cats[i+1]))
                for toc in cats[i+1:]:
                    train_edges_cc.append([cats[i],toc])
                    train_edges_cc_set.add((cats[i],toc))

    
    # print("len(tree_train_edges_cc)=", len(tree_train_edges_cc))
    # with open("tree_directed_cora_tree_train_edges_cc.txt",'w') as fout:
    #     for e in tree_train_edges_cc:
    #         fout.write(str(e[0])+"\t"+str(e[1])+"\n")
                    
    print("len(train_edges_pc)=", len(train_edges_pc))
    print("len(train_edges_cc)=", len(train_edges_cc))
    # use the distinct c-c relations
    print("len(train_edges_cc_set)=", len(train_edges_cc_set))
    train_edges_cc = list(train_edges_cc_set)


    return graph_data, torch.LongTensor(all_edges), torch.LongTensor(train_edges), torch.LongTensor(valid_edges), torch.LongTensor(test_edges),\
           torch.LongTensor(train_edges_pc),torch.LongTensor(train_edges_cc), dict_node_type, dict_cat_idx,catidx_type


def get_degrees (graph_data):
    print ("\n ...get_degrees...")
    edge_list = list(graph_data["all"].edges)
    node_list = list(graph_data["all"].nodes)
    node_degree ={}
    node_indegree ={}
    node_outdegree ={}
    for e in edge_list:
        fr, to = e[0], e[1]
        node_outdegree[fr]= node_outdegree.get(fr,0) +1
        node_indegree[to] = node_indegree.get(to, 0) + 1
        node_degree[fr] = node_degree.get(fr, 0) + 1
        node_degree[to] = node_degree.get(to, 0) + 1
    print ("len(node_indegree)=", len(node_indegree))
    print("len(node_outdegree)=", len(node_outdegree))
    print("len(node_degree)=", len(node_degree))
    return node_indegree, node_outdegree, node_degree

def get_norm_degree(best_emb, node_degree, nodeset=None):
    assert best_emb.size(0) == len(node_degree)
    print ("best_emb.size()=",best_emb.size())
    
    ## map the embedding to lorentz space,. In fact, they are same
    # best_emb_norm = torch.norm(best_emb,dim=-1,keepdim=True) # B*dim -> B*1
    # beta=1.0
    # best_emb_norm = best_emb_norm + beta
    # best_emb_norm = torch.sqrt(best_emb_norm)
    # best_emb = torch.cat((best_emb_norm,best_emb),1)
    

    norm_list= torch.norm(best_emb,dim=-1).tolist() # tensor to list
    degree_list =[]
    for idx in range(len(norm_list)): # dict to list
        degree_list.append(node_degree[idx])
    if nodeset is not None:
        norm_list_filtered = []
        degree_list_filtered = []
        for node in nodeset:
            norm_list_filtered.append(norm_list[node])
            degree_list_filtered.append(degree_list[node])
        norm_list = norm_list_filtered
        degree_list = degree_list_filtered


    return norm_list, degree_list

def get_lp_neg_edges(graph_data, test_edges, args):
    '''
    get the same number of negative edges as test_edges
    Args:
        graph_data:
        test_edges:

    Returns:

    '''

    neg_edge_list=[]
    e_size = test_edges.size(0)
    n_nodes = args.n_nodes
    for i in range (e_size):
        fr = test_edges[i,0]
        while True:
            # neg_node = random.randint(0, n_nodes-1)
            neg_node = random.randrange(0, n_nodes)
            if not graph_data["all"].has_edge(fr,neg_node):
                break
        neg_edge_list.append((fr, neg_node))


    neg_edge_tensor = torch.LongTensor(neg_edge_list)
    assert neg_edge_tensor.size() == test_edges.size()
    return neg_edge_tensor



def get_noderec_cases(graph_data, test_edges, args, eval_neg=99):
    noderec_cases_list = []
    e_size = test_edges.size(0)
    n_nodes = args.n_nodes
    # noderec_size= min(e_size,100000)
    for i in range(e_size):
        fr = test_edges[i, 0]
        to = test_edges[i, 1]
        neg_nodeset = set()
        while len(neg_nodeset) < eval_neg:
            # neg_node = random.randint(0, n_nodes-1)
            neg_node = random.randrange(0, n_nodes)
            if not graph_data["all"].has_edge(fr, neg_node):
                neg_nodeset.add(neg_node)
        neg_nodelist =list(neg_nodeset)
        nodelist =[fr,to]
        nodelist.extend(neg_nodelist)
        assert len(nodelist) == (eval_neg + 2)
        noderec_cases_list.append(nodelist)


    noderec_tensor = torch.LongTensor(noderec_cases_list)
    # assert noderec_tensor.size(0) == test_edges.size(0)
    print ("noderec_tensor.size()= ", noderec_tensor.size())
    return noderec_tensor







