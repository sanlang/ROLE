import argparse

def parser_flags():
    parser = argparse.ArgumentParser(
        description="Graph Embedding"
    )
    parser.add_argument(
        "--dataset", default="protein", choices=["protein", "wikipedia", "blogcatalog", "flickr",
                                                 "wikivote", "directed_cora","directed_epinions",'directed_p2p','directed_hephy',
                                                 'directed_flickr','directed_pokec','directed_livejournal',
                                                 "tree_directed_cora", "tree_directed_nyc",  "tree_directed_tky",  "tree_directed_gw_hs",
                                                 "tree_directed_email",
                                                 'directed_cora_new_split','directed_hephy_new_split'],
        help="Graph dataset"
    )
    parser.add_argument(
        "--model", default="GraphEmb", choices=["GraphEmb", "ME", "Dot", "Node2vec","Node2lv", "Node2vec", 'App',
                                                'GraphEmb_Di_Node2lv_Rot','GraphEmb_Di_Node2lv_Matrix',
                                                "GraphEmb_Di_Node2lv_Rot_Each",'GraphEmb_Di_Node2lv_Rot_Each_to','GraphEmb_Di_Node2lv_Rot_Each2',
                                                'GraphEmb_Di_Node2lv_Add','GraphEmb_Di_Node2lv_Add_to',
                                                'Poincare_acl20',
                                                'RotatE','RotatE_Each','ModE','TransE','HAKE',
                                                'Dancar','Dancar_no_bias','Dancar_node2lv','EuclDisk','HypDisk',
                                                'HypDisk_Poincare_nips19_radam',
                                                'Gravity','ATP',
                                                'Poincare_nips17','Poincare_nips17_euclidean','Poincare_nips17_lorentz',
                                                'Poincare_nips19_radam',
                                                'GraphEmb_Di_Node2lv_Rot_Hetero','GraphEmb_Di_Node2lv_Hetero','GraphEmb_Di_Node2lv_Rot_Each_Hetero',
                                                'GraphEmb_Di_Node2lv_Rot_Hetero_CC',
                                                'GraphEmb_Di_RotH',
                                                'GraphEmb_Di_Node2lv_Rot_Hyperbolic','GraphEmb_Di_Node2lv_Rot_ME',
                                                'GraphEmb_Di_Node2lv_Rot_wo'
                                                ], help="Graph embedding model"
    )
    # parser.add_argument(
    #     "--regularizer", choices=["N3", "F2"], default="N3", help="Regularizer"
    # )
    parser.add_argument(
        "--reg", default=0, type=float, help="Regularization weight"
    )


    parser.add_argument(
        "--weight_decay", default=0, type=float, help="Regularization weight, l2 regularization strength"
    )

    parser.add_argument(
        "--beta", default=1.0, type=float, help="beta for squared lorentz distance"
    )
    parser.add_argument(
        "--optimizer", choices=["Adagrad", "Adam", "RiemannianAdam","RiemannianSGD"], default="Adam",
        help="Optimizer"
    )
    parser.add_argument(
        "--max_epochs", default=200, type=int, help="Maximum number of epochs to train for"
    )
    parser.add_argument(
        "--min_epochs", default=5, type=int, help="Minimum number of epochs to train for"
    )
    parser.add_argument(
        "--patience", default=10, type=int, help="Number of epochs before early stopping"
    )
    parser.add_argument(
        "--valid", default=3, type=float, help="Number of epochs before validation"
    )
    parser.add_argument(
        "--rank", default=128, type=int, help="Embedding dimension"
    )

    parser.add_argument(
        "--num_embedding", default=1, type=int, help="number of embedding: 1 or 2"
    )
    parser.add_argument(
        "--batch_size", default=1024, type=int, help="Batch size"
    )
    parser.add_argument(
        "--neg_sample_size", default=5, type=int, help="Negative sample size, -1 to not use negative sampling"
    )
    parser.add_argument(
        "--dropout", default=0, type=float, help="Dropout rate"
    )
    parser.add_argument(
        "--init_size", default=1e-3, type=float, help="Initial embeddings' scale"
    )
    parser.add_argument(
        "--learning_rate", default=1e-2, type=float, help="Learning rate"
    )
    parser.add_argument(
        "--gamma", default=0, type=float, help="Margin for distance-based losses"
    )

    parser.add_argument(
        "--tree_w", default=0.0, type=float, help="the weight of node-category weight"
    )

    parser.add_argument(
        "--bias", default="constant", type=str, choices=["constant", "learn", "none", "learn_fr", "learn_to"], help="Bias type (none for no bias)"
    )

    parser.add_argument(
        "--cuda", default= 0, type= int,
        help="which cuda device to use (-1 for cpu training)"
    )

    parser.add_argument(
        "--log-freq", default= 1, type= int,
        help='how often to compute print train/val metrics (in epochs)'
    )

    parser.add_argument(
        "--eval-freq", default= 1, type= int,
        help='how often to compute val metrics (in epochs)'
    )

    parser.add_argument(
        "--lr_reduce_freq", default= None, type= int,
        help='reduce lr every lr-reduce-freq or None to keep lr constant'
    )
    parser.add_argument(
        "--seed", default= 42, type= int,
        help="seed for training"
    )

    parser.add_argument(
        "--multi_c", action="store_true", help="Multiple curvatures per relation"
    )

    parser.add_argument(
        "--save", default=0, type=int, help="1 to save model and logs and 0 otherwise"
    )

    parser.add_argument(
        '--visual', action='store_true',help='plot the visualization of embeddings'
    )

    parser.add_argument(
        '--debug', action='store_true',help='Print debugging output'
    )

    parser.add_argument(
        '--directed', dest='directed', action='store_true',help='Graph is (un)directed. Default is undirected.'
    )

    parser.add_argument(
        '--weighted', dest='weighted', action='store_true',help='Boolean specifying (un)weighted. Default is unweighted.'
    )

    parser.add_argument(
        '--hetero', dest='hetero', action='store_true',help='hetero network with tree structure. Default is not.'
    )

    parser.add_argument(
        '--pre_train', dest='pre_train', action='store_true', help='use pre_train embeddings (deepwalk/node2vec). Defualt it not'
    )

    parser.add_argument(
        '--pre_train_file', type=str, help='use pre_train embeddings (deepwalk/node2vec).'
    )

    parser.add_argument(
        '--use_rw', dest='use_rw', action='store_true',help='use random walk. Defualt it not'
    )

    parser.add_argument(
        '--generate_rw', dest='generate_rw', action='store_true',help='generate the random walk. Defualt it not'
    )

    parser.add_argument(
        '--num_walks', dest='num_walks', default=0, type=int,help='number of random walk sequence. Defualt it 10'
    )

    parser.add_argument(
        '--walk_length', dest='walk_length', default=0, type=int,help='length for random walk sequence. Defualt it 80'
    )

    parser.add_argument(
        '--window_size', dest='window_size', default=0, type=int,help='window_size for random walk sequence. Defualt it 5'
    )

    parser.add_argument(
        '--max_eval_edges_count', default=100000, type=int,
        help='the max_eval_edges_count, to reduce eval time'
    )

    return parser