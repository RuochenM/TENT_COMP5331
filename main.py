import numpy as np
import random
import torch
import json
from collections import defaultdict
import torch.optim as optim
import torch.nn as nn
import time
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from model.backbones import *
from model.loss import *
from utils.args import *
from utils.utils import *
from utils.dataset import *


if __name__ == '__main__':
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)

    loss_f = nn.CrossEntropyLoss()

    fine_tune_steps = 20
    fine_tune_lr = 0.1

    results=defaultdict(dict)
    dataset = 'cora-full'
    N_set=[5,10]
    K_set=[3,5]

    for N in N_set:
        for K in K_set:
            
            adj_sparse, features, labels, idx_train, idx_val, idx_test, n1s, n2s, class_train_dict, class_test_dict, class_valid_dict = load_data_pretrain(dataset)

            model = GCN_dense(nfeat=args.hidden1,
                            nhid=args.hidden2,
                            nclass=labels.max().item() + 1,
                            dropout=args.dropout)

            GCN_model=GCN_emb(nfeat=features.shape[1],
                        nhid=args.hidden1,
                        nclass=labels.max().item() + 1,
                        dropout=args.dropout)

            classifier = Linear(args.hidden1, labels.max().item() + 1)

            optimizer = optim.Adam([{'params': model.parameters()}, {'params': classifier.parameters()},{'params': GCN_model.parameters()}],
                                lr=args.lr, weight_decay=args.weight_decay)

