
# 2020.09.21 plot the visualization of embeddings
import time
import random
import numpy as np
import getopt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import seaborn as sns
import operator


def save_plot_powerlaw(tdict, filename):
    '''
    show the distribution of tdict, eg. nodeid_degree
    Args:
        tdict:
        filename:

    Returns:

    '''
    d = np.array(list(tdict.values()))
    unique, counts = np.unique(d, return_counts=True)
    fig = plt.figure()
    ax = plt.gca()
    ax.set_xscale('log')
    ax.scatter(unique, counts, c='b', marker='o', s=15)
    ax.set_yscale('log')
    ax.tick_params(labelsize=16)
    plt.ylabel('Count', fontsize=20)
    plt.xlabel('Degree', fontsize=20)
    ax.xaxis.set_label_coords(0.5, -0.07) # (x,y) is the relative position
    # ax.yaxis.set_label_coords(0.0025, 0.5)
    plt.savefig(filename + ".pdf")
    print(filename + ".pdf")
    # plt.show()

    # freq_cnt = np.asarray((unique, counts)).T
    # # print (freq_cnt)
    # np.save(self.result_dir + filename, freq_cnt)


def save_plot_points(xlist, ylist, filename, top_xlist=None, top_ylist=None):
    '''
    show the points (x,y)
    Args:
        self:
        xlist: L2-norm
        ylist: degree
        top_xlist:
        top_ylist:
        filename:

    Returns:

    '''
    xarray = np.array(xlist)
    yarray = np.array(ylist)
    assert xarray.shape == yarray.shape

    fig = plt.figure()
    ax = plt.gca()

    # ax.set_xlim([0.8,1.01])

    # ax.scatter(yarray,xarray,c='b',marker='o',s=10)

    # Plot the top degree nodes
    if top_xlist and top_ylist:
        top_xarray = np.array(top_xlist)
        top_yarray = np.array(top_ylist)
        print(len(top_xlist), len(top_ylist))
        ax.scatter(xarray, yarray, c='b', marker='o', s=5, facecolors='none', label='Node')
        ax.scatter(top_xarray,top_yarray,c='r',marker='s',s=20,label="Category")
        ax.legend(fontsize=20)
    else:
        ax.scatter(xarray, yarray, c='b', marker='o', facecolors='none',s=5)

    ## log-scale
    # ax.set_xscale('log')
    ax.set_yscale('log')
    plt.ylabel('Degree', fontsize=20)
    plt.xlabel('L2 norm', fontsize=20, x=0.5, y=10)
    
    ax.tick_params(labelsize=16)
    ax.xaxis.set_label_coords(0.5, -0.08)
    plt.savefig( filename + ".pdf")
    plt.savefig(filename + ".png")
    print( filename + ".pdf")
    plt.show()

def save_plot_2D_embs_degree(embs, loc_degree, Train_Node_set, filename):
    '''
    2-D embeddings, group by degree
    Args:
        embs: embeddings, tensor
        loc_degree: the degree of each node
        filename: the output file

    Returns:

    '''
    print("...\n start save_plot_2D_embs_degree()...")
    assert embs.size(1)==2
    if Train_Node_set==None:
        node_sets = set(range(embs.size(0)))
    else:
        node_sets= Train_Node_set

    h_degree_list = []
    h_norm_list = []
    m_degree_list = []
    m_norm_list = []
    l_degree_list = []
    l_norm_list = []

    for node in node_sets:
        if loc_degree.get(node, 0) > 200:
            h_degree_list.append(embs[node][0])
            h_norm_list.append(embs[node][1])
        elif loc_degree.get(node, 0) > 50:
            m_degree_list.append(embs[node][0])
            m_norm_list.append(embs[node][1])
        else:
            l_degree_list.append(embs[node][0])
            l_norm_list.append(embs[node][1])

    fig = plt.figure()
    ax = plt.gca()
    plt.axis('equal')
    # ax.set_xlim([-3.25, 3.25])
    # ax.set_ylim([-3.25, 3.25])

    h_xarray = np.array(h_norm_list)
    h_yarray = np.array(h_degree_list)
    m_xarray = np.array(m_norm_list)
    m_yarray = np.array(m_degree_list)
    l_xarray = np.array(l_norm_list)
    l_yarray = np.array(l_degree_list)

    ax.scatter(l_xarray, l_yarray, c='g', marker='o', s=2, alpha=1.0, label="$Degree < 50$")
    ax.scatter(m_xarray, m_yarray, c='b', marker='o', s=6, alpha=1.0, label="$50 \leq Degree \leq 500$")
    ax.scatter(h_xarray, h_yarray, c='r', marker='o', s=12, alpha=1.0, label='$Degree > 500$')

    ax.legend(fontsize=16)
    ax.tick_params(labelsize=14)
    # ax.xaxis.set_label_coords(0.5, -0.06)

    plt.savefig(filename + ".pdf")
    print (filename + ".pdf")
    plt.show()

# 2020.04.17 depict the embedding with groups
def save_plot_2D_embs_group(embs, cat_embs,Train_Node_set, filename):
    print("...\nsave_plot_2D_embs_group()...")
    assert embs.size(1) == 2

    node_x_list=[]
    node_y_list = []
    c_x_list = []
    c_y_list = []
    node_len= embs.size(0)
    c_len=cat_embs.size(0)

    if Train_Node_set == None:
        node_sets = set(range(embs.size(0)))
    else:
        node_sets = Train_Node_set
    c_sets = set(range(c_len))



    for node in node_sets:
        node_x_list.append(embs[node][0])
        node_y_list.append(embs[node][1])
    for c in c_sets:
        c_x_list.append(embs[c][0])
        c_y_list.append(embs[c][1])
    fig = plt.figure()
    ax = plt.gca()
    plt.axis('equal')
    # ax.set_xlim([-3.25, 3.25])
    # ax.set_ylim([-3.25, 3.25])

    c_xarray = np.array(c_x_list)
    c_yarray = np.array(c_y_list)

    n_xarray = np.array(node_x_list)
    n_yarray = np.array(node_y_list)

    ax.scatter(n_xarray, n_yarray, c='c', marker='o', s=4, alpha=1.0, label="Node")
    ax.scatter(c_xarray, c_yarray, c='r', marker='s', s=12, alpha=1.0, label="Category")

    ax.legend(fontsize=18)
    ax.tick_params(labelsize=16)

    plt.savefig(filename + ".pdf")
    print (filename + ".pdf")

def save_plot_2D_embs_group_multi(embs, cat_embs,catidx_type,Train_Node_set, filename):
    '''
    2020.09.22
    Args:
        embs:
        cat_embs:
        filename:

    Returns:

    '''
    print("...\nsave_plot_2D_embs_group()...")
    assert embs.size(1) == 2

    node_x_list=[]
    node_y_list = []
    # c_x_list = []
    # c_y_list = []
    node_len= embs.size(0)
    c_len=cat_embs.size(0)

    if Train_Node_set == None:
        node_sets = set(range(embs.size(0)))
    else:
        node_sets = Train_Node_set
    c_sets = set(range(c_len))

    for node in node_sets:
        node_x_list.append(embs[node][0])
        node_y_list.append(embs[node][1])

    type_list= ["c1", "c2","c3"]
    c1_x_list = []
    c1_y_list = []

    c2_x_list = []
    c2_y_list = []

    c3_x_list = []
    c3_y_list = []


    for c in c_sets:
        # c_x_list.append(embs[c][0])
        # c_y_list.append(embs[c][1])
        type =catidx_type[c]
        if type=="c1":
            c1_x_list.append(embs[c][0])
            c1_y_list.append(embs[c][1])
        elif type=='c2':
            c2_x_list.append(embs[c][0])
            c2_y_list.append(embs[c][1])
        elif type == 'c3':
            c3_x_list.append(embs[c][0])
            c3_y_list.append(embs[c][1])
        else :
            raise ValueError(
                "ValueError type: {0}".format(type)
            )

    fig = plt.figure()
    ax = plt.gca()
    plt.axis('equal')
    # ax.set_xlim([-3.25, 3.25])
    # ax.set_ylim([-3.25, 3.25])

    n_xarray = np.array(node_x_list)
    n_yarray = np.array(node_y_list)

    c1_xarray = np.array(c1_x_list)
    c1_yarray = np.array(c1_y_list)
    c2_xarray = np.array(c2_x_list)
    c2_yarray = np.array(c2_y_list)
    c3_xarray = np.array(c3_x_list)
    c3_yarray = np.array(c3_y_list)



    ax.scatter(n_xarray, n_yarray, c='c', marker='o', s=4, alpha=1.0, label="Node")
    ax.scatter(c1_xarray, c1_yarray, c='r', marker='s', s=20, alpha=1.0, label="Category-1")
    ax.scatter(c2_xarray, c2_yarray, c='b', marker='*', s=12, alpha=1.0, label="Category-2")
    ax.scatter(c3_xarray, c3_yarray, c='k', marker='^', s=8, alpha=1.0, label="Category-3")


    ax.legend(fontsize=18)
    ax.tick_params(labelsize=16)

    plt.savefig(filename + ".pdf")
    print (filename + ".pdf")
    plt.show()





