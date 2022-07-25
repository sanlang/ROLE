"""Train Knowledge Graph embeddings for link prediction."""

import argparse
import json
import logging
import os
import time
import datetime
import sys
import torch
import torch.optim
import numpy as np
import random

from datetime import datetime



import models
# import optimizers.regularizers as regularizers
from models.GraphEmb import ME, Dot, GraphEmb
from utils import dataset
from utils import visual
import optimizers
import train_test
from parser_define import parser_flags

## 2020.08.03 set the enviroment variables
os.environ["LOG_DIR"] = os.getcwd() + "/logs"
os.environ["DATA_PATH"] = os.getcwd() + "/dataset"


def train(args):
    CUDA_LAUNCH_BLOCKING = 1
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.max_epochs if not args.patience else int(args.patience)

    log_level = logging.DEBUG if args.debug else logging.INFO
    log = logging.getLogger('Node2LV')
    logging.basicConfig(level=log_level, format='%(message)s', stream=sys.stdout)

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    # if args.save:
    #     if not args.save_dir:
    #         dt = datetime.datetime.now()
    #         date = f"{dt.year}_{dt.month}_{dt.day}"
    #         models_dir = os.path.join(os.environ['LOG_DIR'], args.model, date)
    #         save_dir = train_test.get_dir_name(models_dir)
    #     else:
    #         save_dir = args.save_dir

    if args.save:
        save_dir = train_test.get_savedir(args.model, args.dataset)

    dataset_path = os.path.join(os.environ["DATA_PATH"], args.dataset)

    ## Load data
    if not args.hetero:
        graph_data, all_edges, train_edges, valid_edges, test_edges = dataset.load_graph_data(args, dataset_path)
    else:
        assert args.dataset in ["tree_directed_cora", "tree_directed_nyc", "tree_directed_tky", "tree_directed_gw_hs", "tree_directed_email"]
        assert args.hetero==True
        graph_data, all_edges, train_edges, valid_edges, test_edges, train_edges_pc, train_edges_cc, dict_node_type, dict_cat_idx, catidx_type = \
            dataset.load_hetero_graph_data(args, dataset_path)
        args.num_cats = len(dict_cat_idx) # set the number of category

    if args.use_rw:
        assert args.window_size>0
        # assert args.directed == True
        input_rw=None
        # input_rw="generate_rw_directed_flickr.txt"  # for directed_flickr dataset
        train_edges = dataset.get_train_edges_rw(args,dataset_path, input_rw)
    if args.generate_rw:
        assert args.use_rw == False
        assert args.num_walks >0
        assert args.walk_length >0
        # assert args.directed == True
        train_edges = dataset.train_edges_generate_rw(args, graph_data["train"])

    args.n_nodes = graph_data["all"].number_of_nodes()
    node_list = list(graph_data["all"].nodes)
    assert args.n_nodes == max(node_list)+1

    # eq_cnt=0
    # for e in train_edges:
    #     if e[0] ==e[1]:
    #         eq_cnt+=1


    ### limit the number of valid_edges and test_edges
    # max_edges_count =100000
    max_edges_count = args.max_eval_edges_count
    if valid_edges.size(0) > max_edges_count:
        valid_edges = valid_edges[:max_edges_count]
    if test_edges.size(0) > max_edges_count:
        test_edges = test_edges[:max_edges_count]

    valid_edges_false = dataset.get_lp_neg_edges(graph_data, valid_edges, args)
    test_edges_false = dataset.get_lp_neg_edges(graph_data, test_edges, args)
    valid_noderec_cases= dataset.get_noderec_cases(graph_data, valid_edges, args)
    test_noderec_cases = dataset.get_noderec_cases(graph_data, test_edges, args)


    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.max_epochs
    ## model and optimizer
    # Model = args.model
    Model = getattr(models, args.model)
    print (args)
    model = Model(args)
    logging.info(str(model))
    # regularizer = getattr(regularizers, args.regularizer)(args.reg)
    if args.optimizer == "RiemannianSGD":
        optimizer = getattr(optimizers, args.optimizer)(model.optim_params(), lr=args.learning_rate)
    else:
        optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.learning_rate,
                                                        weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(0.5)
    )
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0:
        print ("os.environ "+str(args.cuda))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        print (args.device)
        # os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
        # model = model.to(args.device)
        model = model.cuda()

        # put data to cuda

        # for x, val in data.items():
        #     if torch.is_tensor(data[x]):
        #         data[x] = data[x].to(args.device)


    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    best_emb_c =None

    # # pretrain the tree
    # pretrain_cnt = 10
    # if args.hetero == True:
    #     print("\n........Start pretrain.......")
    #     for epoch in range(pretrain_cnt):
    #         train_loss_rel_pc = train_test.train_emb_epoch_rel_pc(model, train_edges_pc, optimizer)
    #         logging.info("train_loss_rel_pc={:.6f}".format(train_loss_rel_pc))
    #     print ("Finish pretrain.......")
    
    
    ## use pre_train
    if args.pre_train:
        model.read_embs(args)

        best_test_metrics = train_test.eval_emb_epoch(model, test_edges, test_edges_false)
        best_test_metrics_noderec = train_test.eval_emb_epoch_noderec(model, test_noderec_cases)
        best_test_metrics.update(best_test_metrics_noderec)  # extend the result of node_rec
        logging.info(
            " ".join(['Epoch: {:04d}'.format(1), train_test.format_metrics(best_test_metrics, 'test')]))
        logging.info("\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ")
        logging.info("{} {} {}".format(str(args.model), str(args.dataset), str(args.pre_train_file)))
        out_strs = '\t'.join(["{:.4f}".format(metric_val) for _, metric_val in best_test_metrics.items()])
        logging.info(out_strs)
        return best_test_metrics
        
        
    for epoch in range(args.max_epochs):
        t = time.time()
        # train_loss = 0.0

        if args.hetero == True:
            # c-c relations
            # train_loss_rel_cc = train_test.train_emb_epoch_rel_cc(model, train_edges_cc, optimizer)
            # logging.info("train_loss_rel_cc={:.6f}".format(train_loss_rel_cc))
            # p-c relations
            train_loss_rel_pc = train_test.train_emb_epoch_rel_pc(model, train_edges_pc, optimizer)
            logging.info("train_loss_rel_pc={:.6f}".format(train_loss_rel_pc))
        train_loss = train_test.train_emb_epoch(model, train_edges, optimizer)
        lr_scheduler.step()
        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   'lr: {}'.format(lr_scheduler.get_last_lr()[0]),
                                   # 'lr: {}'.format(lr_scheduler.get_lr()[0]),
                                   'train_loss: {:.6f}'.format(train_loss),
                                   'time: {:.4f}s'.format(time.time() - t)
                                   ]))

        if (epoch + 1) % args.eval_freq == 0:
            val_metrics =train_test.eval_emb_epoch(model, valid_edges, valid_edges_false)
            # test_metrics = train_test.eval_emb_epoch(model, test_edges, test_edges_false)
            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), train_test.format_metrics(val_metrics, 'val')]))
            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = train_test.eval_emb_epoch(model, test_edges, test_edges_false)
                best_test_metrics_noderec = train_test.eval_emb_epoch_noderec(model,test_noderec_cases)
                best_test_metrics.update(best_test_metrics_noderec) # extend the result of node_rec
                logging.info(
                    " ".join(['Epoch: {:04d}'.format(epoch + 1), train_test.format_metrics(best_test_metrics, 'test')]))
                best_emb = model.get_embeddings().cpu()
                if args.hetero:
                    best_emb_c = model.get_c_embeddings().cpu()
                if args.save:
                    np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter == args.patience and epoch > args.min_epochs:
                    logging.info("Early stopping")
                    break
    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    if not best_test_metrics:
        best_test_metrics = train_test.eval_emb_epoch(model, test_edges, test_edges_false)
        best_test_metrics_noderec = train_test.eval_emb_epoch_noderec(model, test_noderec_cases)
        best_test_metrics.update(best_test_metrics_noderec)
    logging.info(" ".join(["Val set results:", train_test.format_metrics(best_val_metrics, 'val')]))
    logging.info(" ".join(["Test set results:", train_test.format_metrics(best_test_metrics, 'test')]))
    logging.info("\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ")
    logging.info("{}  {}".format(str(args.model), str(args.dataset)))
    out_strs = '\t'.join(["{:.4f}".format(metric_val) for _, metric_val in best_test_metrics.items()])
    logging.info( out_strs )

    if args.visual:

        ## plot the degree
        node_indegree, node_outdegree, node_degree = dataset.get_degrees(graph_data)
        fig_dir = "visual_figs/"+str(args.dataset)
        
        embed_file= fig_dir + "_" + str(args.model) + '_embeddings.npy'
        np.save(embed_file, best_emb.cpu().detach().numpy())

        if args.hetero:
            embed_file = fig_dir + "_" + str(args.model) + '_c_embeddings.npy'
            np.save(embed_file, best_emb_c.cpu().detach().numpy())
            
        
        # save node_degree
        node_degree_file = fig_dir + "_node_degree.txt"
        with open(node_degree_file, 'w') as fout:
            for k,v in node_degree.items():
                fout.write(str(k)+"\t"+str(v)+"\n")
            print ("write node_degree.txt  ", len(node_degree))
        
        
        fig_indegree= fig_dir+"_indegree"
        fig_outdegree = fig_dir + "_outdegree"
        fig_degree = fig_dir + "_degree"
        assert len(node_degree) == args.n_nodes
        visual.save_plot_powerlaw(node_indegree, fig_indegree)
        visual.save_plot_powerlaw(node_outdegree, fig_outdegree)
        visual.save_plot_powerlaw(node_degree, fig_degree)

        Train_Node_list = list(graph_data["train"].nodes)
        Train_Node_set = set(Train_Node_list)

        if args.rank==2:
            assert model.embedding_dim==2
            D2_fig = fig_dir + "_" + str(args.model) + "_D2"
            visual.save_plot_2D_embs_degree(best_emb, node_degree, Train_Node_set, filename=D2_fig)

            if args.hetero:
                D2_hetero_fig = fig_dir + "_" + str(args.model) + "_D2_hetero"
                visual.save_plot_2D_embs_group(best_emb, best_emb_c,Train_Node_set,filename=D2_hetero_fig)

                D2_hetero_multi_c_fig = fig_dir + "_" + str(args.model) + "_D2_hetero_multi_c"
                visual.save_plot_2D_embs_group_multi(best_emb, best_emb_c, catidx_type, Train_Node_set, filename=D2_hetero_multi_c_fig)
        else:


            ## plot the degree and l2-norm

            l2_norm_list, degree_list = dataset.get_norm_degree(best_emb, node_degree, nodeset=Train_Node_set)
            # l2_norm_list, degree_list = dataset.get_norm_degree(best_emb, node_degree)
            norm_degree_fig = fig_dir + "_"+str(args.model) +"_norm_degree"
            visual.save_plot_points(l2_norm_list,degree_list,norm_degree_fig)

            ## plot the hetero: degree and l2-norm
            if args.hetero:
                c_degree={} # category degree
                edges_pc = train_edges_pc.tolist() # tensor to list
                for e in edges_pc:
                    c= e[1]
                    c_degree[c] = c_degree.get(c, 0) + 1

                c_l2_norm_list, c_degree_list = dataset.get_norm_degree(best_emb_c, c_degree)
                hetero_degree_fig = fig_dir + "_" + str(args.model) + "_hetero_norm_degree"
                visual.save_plot_points(l2_norm_list, degree_list, hetero_degree_fig, top_xlist=c_l2_norm_list, top_ylist=c_degree_list )

    if args.save:
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        logging.info(f"Saved model in {save_dir}")

    return best_test_metrics


if __name__ == "__main__":
    start_time = time.time()
    parser = parser_flags()
    args = parser.parse_args()
    train (args)
    print (args)
    now = datetime.now()
    # current_time = now.strftime("%H:%M:%S")
    print("Current Time =", now)
    # now = datetime.datetime.now()
    # print("Time :", now)
    print ('Total run time: {:.2f} minute'.format((time.time() - start_time)/60.0))
    logging.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n")


