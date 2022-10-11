import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing
from .utils import *
import json
from collections import defaultdict

valid_num_dic = {'Amazon_eletronics': 36, 'dblp': 27}

def load_data_pretrain(dataset_source):
    class_list_train,class_list_valid,class_list_test=json.load(open('./dataset/{}_class_split.json'.format(dataset_source)))
    if dataset_source in valid_num_dic.keys():

        n1s = []
        n2s = []
        for line in open("./dataset/{}_network".format(dataset_source)):
            n1, n2 = line.strip().split('\t')
            n1s.append(int(n1))
            n2s.append(int(n2))

        data_train = sio.loadmat("./dataset/{}_train.mat".format(dataset_source))
        data_test = sio.loadmat("./dataset/{}_test.mat".format(dataset_source))

        num_nodes = max(max(n1s),max(n2s)) + 1
        labels = np.zeros((num_nodes,1))
        labels[data_train['Index']] = data_train["Label"]
        labels[data_test['Index']] = data_test["Label"]

        features = np.zeros((num_nodes,data_train["Attributes"].shape[1]))
        features[data_train['Index']] = data_train["Attributes"].toarray()
        features[data_test['Index']] = data_test["Attributes"].toarray()


        print('nodes num',num_nodes)
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                                shape=(num_nodes, num_nodes))    

        class_list = []
        for cla in labels:
            if cla[0] not in class_list:
                class_list.append(cla[0])  # unsorted

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels):
            id_by_class[cla[0]].append(id)

        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)

        adj = normalize(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        adj = sparse_mx_to_torch_sparse_tensor(adj)

    elif dataset_source=='cora-full':
        adj, features, labels, node_names, attr_names, class_names, metadata=load_npz_to_sparse_graph('./dataset/cora_full.npz')
             
        sparse_mx = adj.tocoo().astype(np.float32)
        indices =np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        
        n1s=indices[0].tolist()
        n2s=indices[1].tolist()
        
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
        
        adj = normalize(adj.tocoo() + sp.eye(adj.shape[0]))
        adj= sparse_mx_to_torch_sparse_tensor(adj)
        features=features.todense()
        features = torch.FloatTensor(features)
        labels=torch.LongTensor(labels).squeeze()
                
            
        class_list =  class_list_train+class_list_valid+class_list_test

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels.numpy().tolist()):
            id_by_class[cla].append(id)
        
    elif dataset_source=='ogbn-arxiv':

        from ogb.nodeproppred import NodePropPredDataset

        dataset = NodePropPredDataset(name = dataset_source)

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, labels = dataset[0] # graph: library-agnostic graph object

        n1s=graph['edge_index'][0]
        n2s=graph['edge_index'][1]

        num_nodes = graph['num_nodes']
        print('nodes num',num_nodes)
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                                shape=(num_nodes, num_nodes))    
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        features=torch.FloatTensor(graph['node_feat'])
        labels=torch.LongTensor(labels).squeeze()

        
        class_list =  class_list_train+class_list_valid+class_list_test

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels.numpy().tolist()):
            id_by_class[cla].append(id)


    idx_train,idx_valid,idx_test=[],[],[]
    for idx_,class_list_ in zip([idx_train,idx_valid,idx_test],[class_list_train,class_list_valid,class_list_test]):
        for class_ in class_list_:
            idx_.extend(id_by_class[class_])

    class_train_dict=defaultdict(list)
    for one in class_list_train:
        for i,label in enumerate(labels.numpy().tolist()):
            if label==one:
                class_train_dict[one].append(i)
    class_valid_dict = defaultdict(list)
    for one in class_list_valid:
        for i, label in enumerate(labels.numpy().tolist()):
            if label == one:
                class_valid_dict[one].append(i)


    class_test_dict = defaultdict(list)
    for one in class_list_test:
        for i, label in enumerate(labels.numpy().tolist()):
            if label == one:
                class_test_dict[one].append(i)
    return adj, features, labels, idx_train, idx_valid, idx_test, n1s, n2s, class_train_dict, class_test_dict, class_valid_dict


def neighborhoods_(adj, n_hops, use_cuda):
    """Returns the n_hops degree adjacency matrix adj."""
    # adj = torch.tensor(adj, dtype=torch.float)
    # adj=adj.to_dense()
    # print(type(adj))
    if use_cuda:
        adj = adj.cuda()
    # hop_adj = power_adj = adj

    # return (adj@(adj.to_dense())+adj).to_dense().cpu().numpy().astype(int)

    hop_adj = adj + torch.sparse.mm(adj, adj)

    hop_adj = hop_adj.to_dense()
    # hop_adj = (hop_adj > 0).to_dense()

    # for i in range(n_hops - 1):
    # power_adj = power_adj @ adj
    # prev_hop_adj = hop_adj
    # hop_adj = hop_adj + power_adj
    # hop_adj = (hop_adj > 0).float()

    hop_adj = hop_adj.cpu().numpy().astype(int)

    return (hop_adj > 0).astype(int)

    # return hop_adj.cpu().numpy().astype(int)

def neighborhoods(adj, n_hops, use_cuda):
    """Returns the n_hops degree adjacency matrix adj."""
    # adj = torch.tensor(adj, dtype=torch.float)
    # adj=adj.to_dense()
    # print(type(adj))
    if n_hops == 1:
        return adj.cpu().numpy().astype(int)

    if use_cuda:
        adj = adj.cuda()
    # hop_adj = power_adj = adj

    # for i in range(n_hops - 1):
    # power_adj = power_adj @ adj
    hop_adj = adj + adj @ adj
    hop_adj = (hop_adj > 0).float()

    np.save(hop_adj.cpu().numpy().astype(int), './neighborhoods_{}.npy'.format(dataset))

    return hop_adj.cpu().numpy().astype(int)