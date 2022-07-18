
import torch
import torch.optim
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import datetime

# def train_emb_inf (model, optimizer, args, graph_data, all_edges, train_edges, valid_edges, test_edges):
# 	pass

def train_emb_epoch(model, train_edges, optimizer, train_batch_size=4096):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss = 0.0

    actual_examples = train_edges[torch.randperm(train_edges.shape[0])]

    with tqdm(total=actual_examples.shape[0],unit='ex') as bar:
        bar.set_description(f'train loss')
        b_begin = 0
        while b_begin < actual_examples.shape[0]:
            input_batch = actual_examples[b_begin:b_begin + train_batch_size].cuda()
            # input_batch = actual_examples[b_begin:b_begin + train_batch_size].to(args.device)

            # gradient step
            optimizer.zero_grad()
            loss = model(input_batch)
            loss.backward()
            # print (loss.item())

            # update parameters
            optimizer.step()
            # optimizer.update_learning_rate()

            b_begin += train_batch_size
            total_loss += loss.data.item()
            bar.update(input_batch.shape[0])
            bar.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss

def train_emb_epoch_rel_pc(model, train_edges_pc, optimizer, train_batch_size=4096):
    ''' 2020.09.08 Train p-c relations '''

    model.train()
    total_loss = 0.0
    rel_pc_weight =model.tree_w

    actual_examples = train_edges_pc[torch.randperm(train_edges_pc.shape[0])]

    with tqdm(total=actual_examples.shape[0],unit='ex') as bar:
        bar.set_description(f'train p-c loss')
        b_begin = 0
        while b_begin < actual_examples.shape[0]:
            input_batch = actual_examples[b_begin:b_begin + train_batch_size].cuda()
            # input_batch = actual_examples[b_begin:b_begin + train_batch_size].to(args.device)

            # gradient step
            optimizer.zero_grad()
            loss = rel_pc_weight* model.rel_pc_forward(input_batch)
            loss.backward()
            # print (loss.item())

            # update parameters
            optimizer.step()
            # optimizer.update_learning_rate()

            b_begin += train_batch_size
            total_loss += loss.data.item()
            bar.update(input_batch.shape[0])
            bar.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss

def train_emb_epoch_rel_cc(model, train_edges_cc, optimizer, train_batch_size=4096):
    ''' 2020.09.13 Train c-c relations '''

    model.train()
    total_loss = 0.0
    rel_pc_weight =model.tree_w

    actual_examples = train_edges_cc[torch.randperm(train_edges_cc.shape[0])]

    with tqdm(total=actual_examples.shape[0],unit='ex') as bar:
        bar.set_description(f'train p-c loss')
        b_begin = 0
        while b_begin < actual_examples.shape[0]:
            input_batch = actual_examples[b_begin:b_begin + train_batch_size].cuda()
            # input_batch = actual_examples[b_begin:b_begin + train_batch_size].to(args.device)

            # gradient step
            optimizer.zero_grad()
            loss = rel_pc_weight* model.rel_cc_forward(input_batch)
            loss.backward()
            # print (loss.item())

            # update parameters
            optimizer.step()
            # optimizer.update_learning_rate()

            b_begin += train_batch_size
            total_loss += loss.data.item()
            bar.update(input_batch.shape[0])
            bar.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss

def eval_emb_epoch(model, eval_edges, eval_edges_false, eval_batch_size=128, k_list=[10,50,100]):
    ''' Epoch operation in evaluation phase
    Note that when eval_edges and eval_edges_false are very big, we need to constraint its size
    '''
    model.eval()
    assert eval_edges.size() == eval_edges_false.size()
    eval_edges = eval_edges.cuda()
    eval_edges_false = eval_edges_false.cuda()
    pos_scores = model.get_edge_score(eval_edges) # (B,1,1)
    neg_scores = model.get_edge_score(eval_edges_false) # (B,1,1)
    pos_scores = pos_scores.squeeze()
    neg_scores = neg_scores.squeeze()
    if pos_scores.is_cuda:
        pos_scores = pos_scores.cpu()
        neg_scores = neg_scores.cpu()
    labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
    preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
    roc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)
    metrics = {'roc': roc, 'ap': ap}
    return metrics


def eval_emb_epoch_noderec(model, eval_noderec_cases, eval_batch_size = 1024, topk=[1,5], metrics=["NDCG","HR"]):
    model.eval()
    # eval_noderec_cases = eval_noderec_cases.cuda()
    total_size = eval_noderec_cases.shape[0]
    noderec_scores = list()
    with tqdm(total=total_size,unit='ex') as bar:
        bar.set_description(f'train loss')
        b_begin = 0
        while b_begin < total_size:
            noderec_batch = eval_noderec_cases[b_begin:b_begin + eval_batch_size].cuda()
            score = model.get_noderec_score(noderec_batch)
            noderec_scores.extend(score.cpu().data.numpy())
            b_begin += eval_batch_size
            bar.update(noderec_batch.shape[0])
            # bar.set_postfix(loss=f'{loss.item():.4f}')
    noderec_scores= np.array(noderec_scores)
    assert noderec_scores.shape[0] == eval_noderec_cases.shape[0]
    assert noderec_scores.shape[1] == eval_noderec_cases.shape[1] -1
    results = evaluate_method(noderec_scores, topk, metrics)
    return results


def evaluate_method(predictions, topk, metrics):
    """
    :param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
    :param topk: top-K values list
    :param metrics: metrics string list
    :return: a result dict, the keys are metrics@topk
    """
    evaluations = dict()
    ## Original implementation
    sort_idx = (-predictions).argsort(axis=1)
    ## Another implementation
    # pred_to_sort = -predictions
    # rand_m = np.random.random(pred_to_sort.shape)
    # sort_idx = np.lexsort((rand_m,pred_to_sort))
    #
    gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1
    for k in topk:
        hit = (gt_rank <= k)
        for metric in metrics:
            key = '{}@{}'.format(metric, k)
            if metric == 'HR':
                evaluations[key] = hit.mean()
            elif metric == 'NDCG':
                evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
            else:
                raise ValueError('Undefined evaluation metric: {}.'.format(metric))
    return evaluations


def format_metrics(metrics, split):
    """Format metric in metric dict for logging."""
    return " ".join(
            ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])

def get_dir_name(models_dir):
    """Gets a directory to save the model.

    If the directory already exists, then append a new integer to the end of
    it. This method is useful so that we don't overwrite existing models
    when launching new jobs.

    Args:
        models_dir: The directory where all the models are.

    Returns:
        The name of a new directory to save the training logs and model weights.
    """
    if not os.path.exists(models_dir):
        save_dir = os.path.join(models_dir, '0')
        os.makedirs(save_dir)
    else:
        existing_dirs = np.array(
                [
                    d
                    for d in os.listdir(models_dir)
                    if os.path.isdir(os.path.join(models_dir, d))
                    ]
        ).astype(np.int)
        if len(existing_dirs) > 0:
            dir_id = str(existing_dirs.max() + 1)
        else:
            dir_id = "1"
        save_dir = os.path.join(models_dir, dir_id)
        os.makedirs(save_dir)
    return save_dir


def get_savedir(model, dataset):
    """Get unique saving directory name."""
    dt = datetime.datetime.now()
    date = dt.strftime("%m_%d")
    save_dir = os.path.join(
        os.environ["LOG_DIR"], date, dataset,
        model + dt.strftime('_%H_%M_%S')
    )
    os.makedirs(save_dir)
    return save_dir





