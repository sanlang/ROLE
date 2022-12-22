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


    if args.save:
        save_dir = train_test.get_savedir(args.model, args.dataset)

    dataset_path = os.path.join(os.environ["DATA_PATH"], args.dataset)

    ## Load data
    if not args.hetero:
        graph_data, all_edges, train_edges, valid_edges, test_edges = dataset.load_graph_data(args, dataset_path)
    else:
        assert args.dataset in ["tree_directed_cora"]
        assert args.hetero==True
        graph_data, all_edges, train_edges, valid_edges, test_edges, train_edges_pc, train_edges_cc, dict_node_type, dict_cat_idx, catidx_type = \
            dataset.load_hetero_graph_data(args, dataset_path)
        args.num_cats = len(dict_cat_idx) # set the number of category


    args.n_nodes = graph_data["all"].number_of_nodes()
    node_list = list(graph_data["all"].nodes)
    assert args.n_nodes == max(node_list)+1


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



    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    best_emb_c =None

        
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


