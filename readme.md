
2022.05.27
Start the revision for TKDE-21

2022.06.21 Re-start the revision again

# Build the Project

local: F:\codes\ROLE\ROLE_TKDE_revision

ssh ices@10.248.19.214
pw: Yyds12345!@#$%
cd /mnt/B/victor_codes/ROLE_TKDE

Remote SSH Interpreter
/home/ices/anaconda2/envs/TOIS_POI/bin/python3.7

conda env list
conda activate TOIS_POI

# Run the code
2022.07.05 Re-start the experiments
Use the V100 server
icess@10.248.19.24

cd /home/ices/victor/ROLE_TKDE_revision
enviroment refer to https://github.com/HazyResearch/KGEmb

conda create --name ROLE python=3.8
conda activate ROLE
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch

## The command lines of ROLE 

python run.py --model GraphEmb_Di_Node2lv_Rot --dataset directed_cora --directed --bias learn
python run.py --model GraphEmb_Di_Node2lv_Rot --dataset directed_hephy --directed --bias learn
python run.py --model GraphEmb_Di_Node2lv_Rot --dataset directed_epinions --directed --bias learn
python run.py --model GraphEmb_Di_Node2lv_Rot --dataset directed_flickr --directed --bias learn


'''
Node2vec: python run.py --model Node2vec --dataset directed_cora --num_embedding 2 --use_rw --window_size 5
PoincareEmb:  python run.py --model Poincare_acl20 --dataset directed_cora --num_embedding 1
SLDE: python run.py --model Node2lv --dataset directed_cora --num_embedding 1
APP:  python run.py --model Dot --dataset directed_cora --directed --num_embedding 2 --use_rw --window_size 5
SLDE-2: python run.py --model Node2lv --dataset directed_cora --directed --num_embedding 2
Dankar: python run.py --model Dancar --dataset directed_cora --directed --num_embedding 2 --bias learn_fr
RotH: python run.py --model GraphEmb_Di_RotH --dataset directed_cora --directed --bias learn
ROLE-P: python run.py --model GraphEmb_Di_Node2lv_Rot_Hyperbolic --dataset directed_cora --directed --bias learn
ROLE: python run.py --model GraphEmb_Di_Node2lv_Rot --dataset directed_cora --directed --bias learn
ROLE+:

use APP
python run.py --model Dot --dataset wikivote --directed --num_embedding 2 --generate_rw --num_walks 10 --walk_length 80 --window_size 5
python run.py --model Dot --dataset directed_flickr --directed --num_embedding 2 --generate_rw --num_walks 10 --walk_length 40 --window_size 5
python run.py --model Dot --dataset directed_flickr --directed --num_embedding 2 --use_rw --window_size 5
'''

### Try larger dataset
2022.07.08

Try the pokec dataset
Nodes	1632803
Edges	30622564


## Run on different GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
model = model.cuda() # 尽量不要使用 model = model.to("cuda:0"), 因为前面已经指定了cuda环境，不需要重复指定了

# Implement the baselines

## Hyp-disk
Hyperbolic Disk Embeddings for Directed Acyclic Graphs, ICML-19

codes: https://github.com/lapras-inc/disk-embedding 
1）run.py，主文件，执行命令改为luigi.run(main_task_cls=RunAll, local_scheduler = True)
2）named_tasks.py，参数文件
3）./models/disk_emb_model_orig.py 欧式空间中的disk embedding类
4）./models/poincare_model.py 非欧空间中的disk embedding类
5）./models/dag_emb_model.py 所有model的基础类


Dankar: 
python run.py --model Dancar --dataset directed_cora --directed --num_embedding 2 --bias learn_fr
0.9181  0.9427  0.6985  0.6985  0.7986  0.8749
python run.py --model Dancar_no_bias --dataset directed_cora --directed --num_embedding 2 --bias none
0.5165  0.4816  0.0200  0.0200  0.0257  0.0309
python run.py --model Dancar_node2lv --dataset directed_cora --directed --num_embedding 2 --bias learn_fr
0.9530  0.9611  0.6988  0.6988  0.8012  0.8793

Based on the dankar, implement the EuclDisk and HypDisk
python run.py --model EuclDisk --dataset directed_cora --directed --num_embedding 1 --bias learn_fr
0.9097  0.9190  0.5642  0.5642  0.6956  0.8034

(Use Adam optimizer)
python run.py --model HypDisk --dataset directed_cora --directed --num_embedding 1 --bias learn_fr
0.8970  0.9135  0.5810  0.5810  0.7098  0.8139
python run.py --model HypDisk --dataset directed_hephy --directed --num_embedding 1 --bias learn_fr  --cuda 1
0.8984  0.9010  0.4479  0.4479  0.6097  0.7475

use RiemannianAdam (which is better than Adam)
python run.py --model HypDisk_Poincare_nips19_radam --dataset directed_cora --directed --num_embedding 1 --bias learn_fr --optimizer RiemannianAdam
0.9461  0.9521  0.6764  0.6764  0.7856  0.8698
python run.py --model HypDisk_Poincare_nips19_radam --dataset directed_hephy --directed --num_embedding 1 --bias learn_fr --optimizer RiemannianAdam
0.9574  0.9536  0.5848  0.5848  0.7483  0.8820

## Gravity
Gravity-Inspired Graph Autoencoders for Directed Link Prediction
https://github.com/deezer/gravity_graph_autoencoders 

The cost in https://github.com/deezer/gravity_graph_autoencoders/blob/master/gravity_gae/optimizer.py
cost=tf.nn.weighted_cross_entropy_with_logits
self.optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)

The self-implemented codes are in models/NestedEmb.py

python run.py --model Gravity --dataset directed_cora --directed --num_embedding 1 --bias learn_fr
0.9029  0.9198  0.5026  0.5026  0.6279  0.7303
python run.py --model Gravity --dataset directed_hephy --directed --num_embedding 1 --bias learn_fr
0.9384  0.9452  0.4641  0.4641  0.6277  0.7625


## ATP
ATP: Directed Graph Embedding with Asymmetric Transitivity Preservation (AAAI-2019)
https://github.com/zhenv5/atp   python
Since ATP is quite different from node embedding, we use the original code to learn the embedding of each node
S is saved at: *_W.pkl
T is saved at: *_H.pkl

Similar to nodevec command
python run.py --model Dot --dataset tree_directed_cora --directed --num_embedding 1 --pre_train --pre_train_file dataset/tree_directed_cora/tree_directed_cora_train_node2vec.emd
Note that need to use the new split dataset

For cora dataset: (Dim=64)
python run.py --model ATP --dataset directed_cora_new_split --directed --num_embedding 2 --bias none --rank 64 --pre_train --pre_train_file dataset/directed_cora_new_split/ATP_directed_cora
0.8619  0.8609  0.3154  0.3154  0.4847  0.6411
For hephy dataset:
python run.py --model ATP --dataset directed_hephy_new_split --directed --num_embedding 2 --bias none --rank 64 --pre_train --pre_train_file dataset/directed_hephy_new_split/ATP_directed_hephy
0.7619  0.6785  0.0919  0.0919  0.1694  0.2442

Try the ROLE for new split datasets
python run.py --model GraphEmb_Di_Node2lv_Rot --dataset directed_cora_new_split --directed --bias learn
0.9835  0.9865  0.8096  0.8096  0.8926  0.9539
python run.py --model GraphEmb_Di_Node2lv_Rot --dataset directed_hephy_new_split --directed --bias learn
0.9910  0.9908  0.7706  0.7706  0.8845  0.9689

The performance of ATP is not so good. 

python run.py --model Dot --dataset directed_cora_new_split --directed --num_embedding 2
0.9082  0.9141  0.5556  0.5556  0.6568  0.7385
(The performance of ATP, which is worse than APP, source and target)


The performance old_split vs new_split (For the different split, similar results)
python run.py --model GraphEmb_Di_Node2lv_Rot --dataset directed_cora --directed --bias learn
0.9665  0.9735  0.7833  0.7833  0.8623  0.9217
python run.py --model GraphEmb_Di_Node2lv_Rot --dataset directed_hephy --directed --bias learn
0.9875  0.9893  0.7678  0.7678  0.8802  0.9636

Use the Dim=128 for ATP
For cora dataset:
python run.py --model ATP --dataset directed_cora_new_split --directed --num_embedding 2 --bias none --rank 128 --pre_train --pre_train_file dataset/directed_cora_new_split/ATP_directed_cora
0.8731  0.8748  0.3562  0.3562  0.5338  0.6963
For hephy dataset:
python run.py --model ATP --dataset directed_hephy_new_split --directed --num_embedding 2 --bias none --rank 128 --pre_train --pre_train_file dataset/directed_hephy_new_split/ATP_directed_hephy
0.7717  0.6891  0.1055  0.1055  0.1836  0.2579
