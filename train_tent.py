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

    Q=10

    fine_tune_steps = 20
    fine_tune_lr = 0.1

    results=defaultdict(dict)

    for dataset in ['cora-full','Amazon_eletronics','dblp','ogbn-arxiv']:

        adj_sparse, features, labels, idx_train, idx_val, idx_test, n1s, n2s, class_train_dict, class_test_dict, class_valid_dict = load_data_pretrain(dataset)

        adj = adj_sparse.to_dense()
        if dataset!='ogbn-arxiv':
            adj=adj.cuda()
        else:
            args.use_cuda=False

        N_set=[5,10]
        K_set=[3,5]

        for N in N_set:
            for K in K_set:
                for repeat in range(5):
                    print('done')
                    print(dataset)
                    print('N={},K={}'.format(N,K))  
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

                    support_labels=torch.zeros(N*K,dtype=torch.long)
                    for i in range(N):
                        support_labels[i * K:(i + 1) * K] = i
                    query_labels=torch.zeros(N*Q,dtype=torch.long)
                    for i in range(N):
                        query_labels[i * Q:(i + 1) * Q] = i

                    if args.use_cuda:
                        model.cuda()
                        features = features.cuda()
                        GCN_model = GCN_model.cuda()
                        adj_sparse = adj_sparse.cuda()
                        labels = labels.cuda()
                        classifier = classifier.cuda()

                        support_labels=support_labels.cuda()
                        query_labels=query_labels.cuda()

                    def pre_train(epoch, N, mode='train'):

                        if mode == 'train':
                            model.train()
                            optimizer.zero_grad()
                        else:
                            model.eval()

                        emb_features=GCN_model(features, adj_sparse)

                        target_idx = []
                        target_graph_adj_and_feat = []
                        support_graph_adj_and_feat = []

                        pos_node_idx = []

                        if mode == 'train':
                            class_dict = class_train_dict
                        elif mode == 'test':
                            class_dict = class_test_dict
                        elif mode=='valid':
                            class_dict = class_valid_dict
                        
                        classes = np.random.choice(list(class_dict.keys()), N, replace=False).tolist()

                        pos_graph_adj_and_feat=[]   
                        for i in classes:
                            # sample from one specific class
                            sampled_idx=np.random.choice(class_dict[i], K+Q, replace=False).tolist()
                            pos_node_idx.extend(sampled_idx[:K])
                            target_idx.extend(sampled_idx[K:])

                            class_pos_idx=sampled_idx[:K]

                            if K==1 and torch.nonzero(adj[class_pos_idx,:]).shape[0]==1:
                                pos_class_graph_adj=adj[class_pos_idx,class_pos_idx].reshape([1,1])
                                pos_graph_feat=emb_features[class_pos_idx]
                            else:
                                pos_graph_neighbors = torch.nonzero(adj[class_pos_idx, :].sum(0)).squeeze()

                                pos_graph_adj = adj[pos_graph_neighbors, :][:, pos_graph_neighbors]

                                pos_class_graph_adj=torch.eye(pos_graph_neighbors.shape[0]+1,dtype=torch.float)

                                pos_class_graph_adj[1:,1:]=pos_graph_adj

                                pos_graph_feat=torch.cat([emb_features[class_pos_idx].mean(0,keepdim=True),emb_features[pos_graph_neighbors]],0)

                            if dataset!='ogbn-arxiv':
                                pos_class_graph_adj=pos_class_graph_adj.cuda()

                            pos_graph_adj_and_feat.append((pos_class_graph_adj, pos_graph_feat))

                        target_graph_adj_and_feat=[]  
                        for node in target_idx:
                            if torch.nonzero(adj[node,:]).shape[0]==1:
                                pos_graph_adj=adj[node,node].reshape([1,1])
                                pos_graph_feat=emb_features[node].unsqueeze(0)
                            else:
                                pos_graph_neighbors = torch.nonzero(adj[node, :]).squeeze()
                                pos_graph_neighbors = torch.nonzero(adj[pos_graph_neighbors, :].sum(0)).squeeze()
                                pos_graph_adj = adj[pos_graph_neighbors, :][:, pos_graph_neighbors]
                                pos_graph_feat = emb_features[pos_graph_neighbors]

                            target_graph_adj_and_feat.append((pos_graph_adj, pos_graph_feat))

                        class_generate_emb=torch.stack([sub[1][0] for sub in pos_graph_adj_and_feat],0).mean(0)

                        parameters=model.generater(class_generate_emb)

                        gc1_parameters=parameters[:(args.hidden1+1)*args.hidden2*2]
                        gc2_parameters=parameters[(args.hidden1+1)*args.hidden2*2:]

                        gc1_w=gc1_parameters[:args.hidden1*args.hidden2*2].reshape([2,args.hidden1,args.hidden2])
                        gc1_b=gc1_parameters[args.hidden1*args.hidden2*2:].reshape([2,args.hidden2])

                        gc2_w=gc2_parameters[:args.hidden2*args.hidden2*2].reshape([2,args.hidden2,args.hidden2])
                        gc2_b=gc2_parameters[args.hidden2*args.hidden2*2:].reshape([2,args.hidden2])

                        model.eval()
                        ori_emb = []
                        for i, one in enumerate(target_graph_adj_and_feat):
                            sub_adj, sub_feat = one[0], one[1]
                            ori_emb.append(model(sub_feat, sub_adj, gc1_w, gc1_b, gc2_w, gc2_b).mean(0))  # .mean(0))

                        target_embs = torch.stack(ori_emb, 0)

                        class_ego_embs=[]
                        for sub_adj, sub_feat in pos_graph_adj_and_feat:
                            class_ego_embs.append(model(sub_feat,sub_adj,gc1_w,gc1_b,gc2_w,gc2_b)[0])
                        class_ego_embs=torch.stack(class_ego_embs,0)

                        target_embs=target_embs.reshape([N,Q,-1]).transpose(0,1)
                        
                        support_features = emb_features[pos_node_idx].reshape([N,K,-1])
                        class_features=support_features.mean(1)
                        taus=[]
                        for j in range(N):
                            taus.append(torch.linalg.norm(support_features[j]-class_features[j],-1).sum(0))
                        taus=torch.stack(taus,0)

                        similarities=[]
                        for j in range(Q):
                            class_contras_loss, similarity=InforNCE_Loss(target_embs[j],dataset,class_ego_embs/taus.unsqueeze(-1),tau=0.5)
                            similarities.append(similarity)

                        loss_supervised=loss_f(classifier(emb_features[idx_train]), labels[idx_train])

                        loss=loss_supervised

                        labels_train=labels[target_idx]
                        for j, class_idx in enumerate(classes[:N]):
                            labels_train[labels_train==class_idx]=j
                            
                        loss+=loss_f(torch.stack(similarities,0).transpose(0,1).reshape([N*Q,-1]), labels_train)
                    
                        acc_train = accuracy(torch.stack(similarities,0).transpose(0,1).reshape([N*Q,-1]), labels_train)

                        if mode=='valid' or mode=='test' or (mode=='train' and epoch%250==249):
                            support_features = l2_normalize(emb_features[pos_node_idx].detach().cpu()).numpy()
                            query_features = l2_normalize(emb_features[target_idx].detach().cpu()).numpy()

                            support_labels=torch.zeros(N*K,dtype=torch.long)
                            for i in range(N):
                                support_labels[i * K:(i + 1) * K] = i

                            query_labels=torch.zeros(N*Q,dtype=torch.long)
                            for i in range(N):
                                query_labels[i * Q:(i + 1) * Q] = i

                            clf = LogisticRegression(penalty='l2',
                                                    random_state=0,
                                                    C=1.0,
                                                    solver='lbfgs',
                                                    max_iter=1000,
                                                    multi_class='multinomial')
                            clf.fit(support_features, support_labels.numpy())
                            query_ys_pred = clf.predict(query_features)

                            acc_train = metrics.accuracy_score(query_labels, query_ys_pred)

                        if mode == 'train':
                            loss.backward()
                            optimizer.step()

                        if epoch % 250 == 249 and mode == 'train':
                            print('Epoch: {:04d}'.format(epoch + 1),
                                'loss_train: {:.4f}'.format(loss.item()),
                                'acc_train: {:.4f}'.format(acc_train.item()))
                        return acc_train.item()

                    # Train model
                    t_total = time.time()
                    best_acc = 0
                    best_valid_acc=0
                    count=0
                    for epoch in range(args.epochs):
                        acc_train=pre_train(epoch, N=N)
                        
                        if  epoch > 0 and epoch % 50 == 0:

                            temp_accs=[]
                            for epoch_test in range(50):
                                temp_accs.append(pre_train(epoch_test, N=N, mode='test'))

                            accs = []

                            for epoch_test in range(50):
                                accs.append(pre_train(epoch_test, N=N if dataset!='ogbn-arxiv' else 5, mode='valid'))

                            valid_acc=np.array(accs).mean(axis=0)
                            print("Epoch: {:04d} Meta-valid_Accuracy: {:.4f}".format(epoch + 1, valid_acc))

                            if valid_acc>best_valid_acc:
                                best_test_accs=temp_accs
                                best_valid_acc=valid_acc
                                count=0
                            else:
                                count+=1
                                if count>=10:
                                    break

                    accs=best_test_accs

                    print('Test Acc',np.array(accs).mean(axis=0))
                    results[dataset]['{}-way {}-shot {}-repeat'.format(N,K,repeat)]=[np.array(accs).mean(axis=0)]

                    json.dump(results[dataset],open('./TENT-result_{}.json'.format(dataset),'w'))


                accs=[]
                for repeat in range(5):
                    accs.append(results[dataset]['{}-way {}-shot {}-repeat'.format(N,K,repeat)][0])


                results[dataset]['{}-way {}-shot'.format(N,K)]=[np.mean(accs)]
                results[dataset]['{}-way {}-shot_print'.format(N,K)]='acc: {:.4f}'.format(np.mean(accs))

                json.dump(results[dataset],open('./TENT-result_{}.json'.format(dataset),'w'))   

                del model

        del adj


