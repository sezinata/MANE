'''

@author: sezin
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import networkx as nx
import numpy as np
import generate_pairs
import random
import gc
import time
from sklearn import preprocessing
from args_parser_Link_Prediction_multiclass_MANE_Attention import get_parser
from collections import OrderedDict
import torch.nn.functional as F


class MANEAttention(nn.Module):
    def __init__(self, n_classes, params, len_common_nodes, embed_freq, batch_size, negative_sampling_size=10):

        super(MANEAttention, self).__init__()
        self.n_embedding = len_common_nodes
        self.embed_freq = embed_freq
        self.num_net = params.nviews
        self.negative_sampling_size = negative_sampling_size
        self.node_embeddings = nn.ModuleList()
        self.neigh_embeddings = nn.ModuleList()
        self.device = params.device
        self.embedding_dim = params.dimensions
        self.n_classes = n_classes
        self.final_emb = torch.randn(self.n_embedding, self.embedding_dim)
        self.nn_linears = nn.ModuleList()

        self.attention = nn.ParameterList([])
        self.b_att = nn.ParameterList([])

        self.batch_size = batch_size

        self.classifier_w = nn.Parameter(
            torch.randn(self.embedding_dim * self.num_net * 2, self.n_classes,
                        requires_grad=True))
        self.classifier_b = nn.Parameter(torch.randn(self.n_classes, requires_grad=True))
        self.b_att = nn.ParameterList([])

        for n_net in range(self.num_net):
            self.node_embeddings.append(nn.Embedding(self.n_embedding, self.embedding_dim))
            self.neigh_embeddings.append(nn.Embedding(self.n_embedding, self.embedding_dim))
            self.attention.append(
                nn.Parameter(torch.randn(self.embedding_dim * self.num_net, requires_grad=True)))
            self.b_att.append(nn.Parameter(torch.randn(1, requires_grad=True)))
            self.nn_linears.append(nn.Linear(self.embedding_dim * self.num_net, 1))

    def forward(self, count, shuffle_indices_nets, nodes_idx_nets, neigh_idx_nets, hyp1, hyp2):
        cost1 = [nn.functional.logsigmoid(torch.bmm(self.neigh_embeddings[i](Variable(torch.LongTensor(
            neigh_idx_nets[i][shuffle_indices_nets[i][count:count + self.batch_size]]).to(self.device))).unsqueeze(
            2).view(
            len(shuffle_indices_nets[i][count:count + self.batch_size]), -1,
            self.embedding_dim), self.node_embeddings[i](Variable(
            torch.LongTensor(nodes_idx_nets[i][shuffle_indices_nets[i][count:count + self.batch_size]]).to(
                self.device))).view(
            len(shuffle_indices_nets[i][count:count + self.batch_size]), -1).unsqueeze(
            2))).squeeze().mean() + nn.functional.logsigmoid(torch.bmm(self.neigh_embeddings[i](
            self.embed_freq.multinomial(
                len(shuffle_indices_nets[i][count:count + self.batch_size]) * self.neigh_embeddings[i](Variable(
                    torch.LongTensor(
                        neigh_idx_nets[i][shuffle_indices_nets[i][count:count + self.batch_size]]).to(
                        self.device))).unsqueeze(
                    2).view(len(shuffle_indices_nets[i][count:count + self.batch_size]), -1,
                            self.embedding_dim).size(1) * self.negative_sampling_size, replacement=True).to(
                self.device)).view(
            len(shuffle_indices_nets[i][count:count + self.batch_size]), -1,
            self.embedding_dim).neg(), self.node_embeddings[i](Variable(
            torch.LongTensor(nodes_idx_nets[i][shuffle_indices_nets[i][count:count + self.batch_size]]).to(
                self.device))).view(
            len(shuffle_indices_nets[i][count:count + self.batch_size]), -1).unsqueeze(2))).squeeze().sum(1).mean(0) for
                 i in range(self.num_net)]

        # First order collaboration:

        cost2 = [[hyp1 * (nn.functional.logsigmoid(torch.bmm(self.node_embeddings[j](Variable(torch.LongTensor(
            nodes_idx_nets[i][shuffle_indices_nets[i][count:count + self.batch_size]]).to(self.device))).unsqueeze(
            2).view(
            len(shuffle_indices_nets[i][count:count + self.batch_size]), -1, self.embedding_dim),
            self.node_embeddings[i](Variable(torch.LongTensor(
                nodes_idx_nets[i][shuffle_indices_nets[i][
                                  count:count + self.batch_size]]).to(self.device))).view(
                len(shuffle_indices_nets[i][
                    count:count + self.batch_size]), -1).unsqueeze(
                2))).squeeze().mean() + nn.functional.logsigmoid(
            torch.bmm(self.node_embeddings[j](self.embed_freq.multinomial(
                len(shuffle_indices_nets[i][count:count + self.batch_size]) * self.node_embeddings[j](Variable(
                    torch.LongTensor(
                        nodes_idx_nets[i][shuffle_indices_nets[i][count:count + self.batch_size]]).to(
                        self.device))).unsqueeze(
                    2).view(len(shuffle_indices_nets[i][count:count + self.batch_size]), -1, self.embedding_dim).size(
                    1) * self.negative_sampling_size,
                replacement=True).to(self.device)).view(len(shuffle_indices_nets[i][count:count + self.batch_size]), -1,
                                                        self.embedding_dim).neg(), self.node_embeddings[i](Variable(
                torch.LongTensor(
                    nodes_idx_nets[i][shuffle_indices_nets[i][count:count + self.batch_size]]).to(self.device))).view(
                len(shuffle_indices_nets[i][count:count + self.batch_size]), -1).unsqueeze(2))).squeeze().sum(1).mean(
            0))
                  for i in range(self.num_net) if i != j] for j in range(self.num_net)]

        # Second order collaboration:

        cost3 = [[hyp2 * (nn.functional.logsigmoid(torch.bmm(self.neigh_embeddings[j](Variable(torch.LongTensor(
            neigh_idx_nets[i][shuffle_indices_nets[i][count:count + self.batch_size]]).to(self.device))).unsqueeze(
            2).view(
            len(shuffle_indices_nets[i][count:count + self.batch_size]), -1, self.embedding_dim),
            self.node_embeddings[i](Variable(torch.LongTensor(
                nodes_idx_nets[i][shuffle_indices_nets[i][
                                  count:count + self.batch_size]]).to(self.device))).view(
                len(shuffle_indices_nets[i][
                    count:count + self.batch_size]), -1).unsqueeze(
                2))).squeeze().mean() + nn.functional.logsigmoid(
            torch.bmm(self.neigh_embeddings[j](self.embed_freq.multinomial(
                len(shuffle_indices_nets[i][count:count + self.batch_size]) * self.neigh_embeddings[j](Variable(
                    torch.LongTensor(
                        neigh_idx_nets[i][shuffle_indices_nets[i][count:count + self.batch_size]]).to(
                        self.device))).unsqueeze(
                    2).view(len(shuffle_indices_nets[i][count:count + self.batch_size]), -1, self.embedding_dim).size(
                    1) * self.negative_sampling_size,
                replacement=True).to(self.device)).view(len(shuffle_indices_nets[i][count:count + self.batch_size]), -1,
                                                        self.embedding_dim).neg(), self.node_embeddings[i](Variable(
                torch.LongTensor(
                    nodes_idx_nets[i][shuffle_indices_nets[i][count:count + self.batch_size]]).to(self.device))).view(
                len(shuffle_indices_nets[i][count:count + self.batch_size]), -1).unsqueeze(2))).squeeze().sum(1).mean(
            0))
                  for i in range(self.num_net) if i != j] for j in range(self.num_net)]

        sum_cost2 = []
        [[sum_cost2.append(j) for j in i] for i in cost2]

        sum_cost3 = []
        [[sum_cost3.append(j) for j in i] for i in cost3]

        return -(torch.mean(torch.stack(cost1)) + sum(sum_cost2) / len(sum_cost2) + sum(sum_cost3) / len(sum_cost3)) / 3

    def supervision_link_multi_class(self, labels, common_nodes, nodeidx, label, hyp_s):

        criterion = nn.CrossEntropyLoss()
        tanh = nn.Tanh()
        allidx = torch.LongTensor(np.arange(self.n_embedding)).to(self.device)

        concat_tensors = self.node_embeddings[0].weight[Variable(allidx)]
        for i_tensor in range(1, self.num_net):
            concat_tensors = (
                torch.cat((concat_tensors, self.node_embeddings[i_tensor].weight[Variable(allidx)]), 1))
        cur_atts = []
        for i_att in range(0, self.num_net):
            cur_atts.append((self.nn_linears[i_att](tanh(
                concat_tensors[Variable(allidx)] * self.attention[i_att].unsqueeze(0).to(self.device) + self.b_att[
                    i_att].to(self.device)))))  # elementwise product

        self.final_emb = torch.bmm(self.node_embeddings[0].weight[Variable(allidx)].unsqueeze(2),
                                   (torch.exp(cur_atts[0]) / sum(torch.exp(cur_atts[i_att]) for i_att
                                                                 in range(self.num_net))).unsqueeze(2))

        for i_tensor in range(1, self.num_net):
            self.final_emb = (torch.cat((self.final_emb, torch.bmm(self.node_embeddings[i_tensor].weight[
                                                                       Variable(allidx)].unsqueeze(2), (
                                                                               torch.exp(cur_atts[i_tensor]) / sum(
                                                                           torch.exp(cur_atts[i_att]) for i_att
                                                                           in range(self.num_net))).unsqueeze(2))), 1))

        embedding = tanh(self.final_emb.squeeze())

        node_embed_dict = dict(zip(common_nodes, embedding.to('cpu')))
        two_nodes = np.column_stack((labels[:, 0], labels[:, 1]))
        two_nodes_tuples = tuple(map(tuple, two_nodes))  # convert pairs to tuple

        ############preparing embedding data based on given labelled pairs#############

        X = [torch.cat((node_embed_dict[each_pair[0]], node_embed_dict[each_pair[1]]), dim=0) for each_pair in
             two_nodes_tuples if (each_pair[0] in node_embed_dict) and (
                     each_pair[1] in node_embed_dict)]

        label = torch.LongTensor(label).to(self.device)

        loss_labels = criterion(
            torch.matmul(torch.stack(X).float().to(self.device), self.classifier_w) + self.classifier_b,
            label)

        cost_sup = hyp_s * loss_labels
        return cost_sup

    '''
    # Clear version of cost1 cost2 and cost3
    cost = []
    for i in range(self.num_net):

        batch_indices = shuffle_indices_nets[i][count:count + self.batch_size]

        nodes_idx = torch.LongTensor(nodesidx_nets[i][batch_indices]).to(self.device)
        node_emb = self.node_embeddings[i](Variable(nodes_idx)).view(len(batch_indices), -1).unsqueeze(2)

        neighs_idx = torch.LongTensor(neighidx_nets[i][batch_indices]).to(self.device)
        neigh_emb = self.neigh_embeddings[i](Variable(neighs_idx)).unsqueeze(2).view(len(batch_indices), -1,
                                                                                    self.embedding_dim)
        loss_positive = nn.functional.logsigmoid(torch.bmm(neigh_emb, node_emb)).squeeze().mean()
        negative_context = self.embed_freq.multinomial(
            len(batch_indices) * neigh_emb.size(1) * self.negative_sampling_size,
            replacement=True).to(self.device)
        negative_context_emb = self.neigh_embeddings[i](negative_context).view(len(batch_indices), -1,
                                                                               self.embedding_dim).neg()
        loss_negative = nn.functional.logsigmoid(torch.bmm(negative_context_emb, node_emb)).squeeze().sum(1).mean(0)
        cost.append(loss_positive + loss_negative)


        for j in range(self.num_net):
            if j != i:
                node_neigh_emb = self.node_embeddings[j](Variable(nodes_idx)).unsqueeze(2).view(len(batch_indices),
                                                                                                -1,
                                                                                                self.embedding_dim)
                loss_positive = nn.functional.logsigmoid(torch.bmm(node_neigh_emb, node_emb)).squeeze().mean()
                negative_context2 = self.embed_freq.multinomial(
                    len(batch_indices) * node_neigh_emb.size(1) * self.negative_sampling_size,
                    replacement=True).to(self.device)

                negative_context_emb2 = self.node_embeddings[j](negative_context2).view(len(batch_indices), -1,
                                                                                        self.embedding_dim).neg()
                loss_negative = nn.functional.logsigmoid(torch.bmm(negative_context_emb2, node_emb)).squeeze().sum(
                    1).mean(0)
                cost.append(hyp1 * (loss_positive + loss_negative))


        for j in range(self.num_net):
            if j != i:
                cross_neighs_idx = torch.LongTensor(
                    neighidx_nets[i][batch_indices]).to(self.device)
                cross_neigh_emb = self.neigh_embeddings[j](Variable(cross_neighs_idx)).unsqueeze(2).view(
                    len(batch_indices), -1,
                    self.embedding_dim)
                loss_positive = nn.functional.logsigmoid(torch.bmm(cross_neigh_emb, node_emb)).squeeze().mean()
                negative_context3 = self.embed_freq.multinomial(
                    len(batch_indices) * cross_neigh_emb.size(1) * self.negative_sampling_size,
                    replacement=True).to(self.device)
                negative_context_emb = self.neigh_embeddings[j](negative_context3).view(len(batch_indices), -1,
                                                                                        self.embedding_dim).neg()
                loss_negative = nn.functional.logsigmoid(torch.bmm(negative_context_emb, node_emb)).squeeze().sum(
                    1).mean(0)
                cost.append(hyp2 * (loss_positive + loss_negative))


    return -sum(cost) / len(cost)
    '''


def read_graphs(current_path, n_views):
    """
        Read graph/network data for each view from an adjlist (from networkx package)

    :param current_path: path for graph data
    :param n_views: number of views
    :return: A list of graphs
    """
    entries = os.listdir(current_path)
    G = []
    if len(entries) != n_views:
        print("WARNING: Number of networks in the folder is not equal to number of views setting.")
    for n_net in range(n_views):
        G.append(nx.read_adjlist(current_path + entries[n_net]))
        print("Network ", (n_net + 1), ": ", entries[n_net])
    return G


def read_word2vec_pairs(current_path, nviews):
    """

    :param current_path: path for two files, one keeps only the node indices, the other keeps only the neighbor node
    indices of already generated pairs (node,neighbor), i.e, node indices and neighbor indices are kept separately.
    method "construct_word2vec_pairs" can be used to obtain these files.
    :E.g.:

      for pairs (9,2) (4,5) (8,6) one file keeps 9 4 8 the other file keeps 2 5 6.

    :param nviews: number of views
    :return: Two lists for all views, each list keeps the node indices of node pairs (node, neigh).
    nodes_idx_nets for node, neigh_idx_nets for neighbor
    """

    nodes_idx_nets = []
    neigh_idx_nets = []

    for n_net in range(nviews):
        neigh_idx_nets.append(np.loadtxt(current_path + "/neighidxPairs_" + str(n_net + 1) + ".txt"))
        nodes_idx_nets.append(np.loadtxt(current_path + "/nodesidxPairs_" + str(n_net + 1) + ".txt"))
    return nodes_idx_nets, neigh_idx_nets


def degree_nodes_common_nodes(G, common_nodes, node2idx):
    """
    Assigns scores for negative sampling distribution
    """
    degrees_idx = dict((node2idx[v], 0) for v in common_nodes)
    multinomial_nodesidx = []
    for node in common_nodes:
        degrees_idx[node2idx[node]] = sum([G[n].degree(node) for n in range(len(G))])
    for node in common_nodes:
        multinomial_nodesidx.append(degrees_idx[node2idx[node]] ** (0.75))

    return multinomial_nodesidx


def read_supervision_link_labels(current_path, splits, repeat):
    """
    Attention version, label file and trains indices and train labels of node pairs are read for link prediction task.
    :param current_path: path for labelled data node_1 node_2 label, train indices and train labels for each fold.
    :param splits: Each fold of cv.
    :param repeat: Repeat value of cross-validation in 1x5 fold cv repeat is set to 1.
    :return: Read values
    """
    train_splits = []
    train_label_splits = []
    labels = np.loadtxt(current_path + "all_label_data.txt", dtype=str)  # node_1 node_2 label

    for split_cur in range(splits):
        split_cur += 1
        train_splits.append(
            np.loadtxt(
                current_path + "train_idx" + repeat + str(split_cur) + ".txt"))
        train_label_splits.append(
            np.loadtxt(
                current_path + "train_label" + repeat + str(split_cur) + ".txt"))

    return train_splits, train_label_splits, labels


params = get_parser().parse_args()


def main():
    """
    Initialize parameters and train
    """

    print(params)

    if torch.cuda.is_available() and not params.cuda:
        print("WARNING: You have a CUDA device, you may try cuda with --cuda")
    device = 'cuda:0' if torch.cuda.is_available() and params.cuda else 'cpu'
    params.device = device
    print("Running on device: ", device)
    G = read_graphs(params.input_graphs + params.dataset, params.nviews)
    common_nodes = sorted(set(G[0]).intersection(*G))
    print('Number of common/core nodes in all networks: ', len(common_nodes))
    node2idx = {n: idx for (idx, n) in enumerate(common_nodes)}
    idx2node = {idx: n for (idx, n) in enumerate(common_nodes)}

    if params.read_pair:

        nodes_idx_nets, neigh_idx_nets = read_word2vec_pairs(params.input_pairs + params.dataset, params.nviews)

    else:
        nodes_idx_nets = []
        neigh_idx_nets = []
        for n_net in range(params.nviews):
            view_id = n_net + 1
            print("View ", view_id)

            nodes_idx, neigh_idx = generate_pairs.construct_word2vec_pairs(G[n_net], view_id, common_nodes, params.p,
                                                                           params.q, params.window_size,
                                                                           params.num_walks,
                                                                           params.walk_length,
                                                                           params.output_pairs + params.dataset,
                                                                           node2idx)

            nodes_idx_nets.append(nodes_idx)
            neigh_idx_nets.append(neigh_idx)

    multinomial_nodes_idx = degree_nodes_common_nodes(G, common_nodes, node2idx)

    embed_freq = Variable(torch.Tensor(multinomial_nodes_idx))

    split_indice = params.cur_split - 1

    repeat = "1"  # for 5x5 fold cv, repeat = 5

    train_splits, train_label_splits, labels = read_supervision_link_labels(params.att_link_label_path + params.dataset,
                                                                            params.nsplits, repeat)
    y = list(map(int, labels[:, 2]))
    train_splits_cur = list(map(int, train_splits[split_indice]))
    ytrain = train_label_splits[split_indice]
    print("Training size: ", len(ytrain))
    lb = preprocessing.LabelBinarizer()
    lb.fit(ytrain)
    one_hot_labels = lb.transform(ytrain)

    train_labels = [labels[int(i)] for i in train_splits[split_indice]]

    labels = np.array(train_labels)

    n_classes = len(np.unique(y))

    if n_classes == 2:
        n_classes = 1
    if (n_classes > 2):  # number of classes, multi-class
        ytrain = np.argmax(one_hot_labels, axis=1)

    n_classes = len(np.unique(train_label_splits[split_indice]))

    if n_classes == 2:
        n_classes = 1
    print("Number of classes: ", n_classes)
    model = MANEAttention(n_classes, params, len(common_nodes), embed_freq, params.batch_size)
    model.to(device)
    epo = 0

    min_pair_length = nodes_idx_nets[0].size
    for n_net in range(params.nviews):
        if min_pair_length > nodes_idx_nets[n_net].size:
            min_pair_length = nodes_idx_nets[n_net].size
    print("Total number of pairs: ", min_pair_length)
    print("Training started! \n")

    epo = 0

    while epo <= params.epochs - 1:
        start_init = time.time()
        epo += 1

        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
        running_loss = 0
        num_batches = 0
        shuffle_indices_nets = []
        fifty = False

        for n_net in range(params.nviews):
            shuffle_indices = [x for x in range(nodes_idx_nets[n_net].size)]
            random.shuffle(shuffle_indices)
            shuffle_indices_nets.append(shuffle_indices)

        model.classifier_w.requires_grad = False
        model.classifier_b.requires_grad = False
        for n_net in range(params.nviews):
            model.attention[n_net].requires_grad = False
            model.nn_linears[n_net].weight.requires_grad = False
            model.b_att[n_net].requires_grad = False
            model.node_embeddings[n_net].weight.requires_grad = True
            model.neigh_embeddings[n_net].weight.requires_grad = True

        optimizer.zero_grad()  # put here
        loss2 = model.supervision_link_multi_class(labels, common_nodes, train_splits_cur, ytrain, params.gamma)
        loss2.backward()
        running_loss += loss2.detach()

        for count in range(0, min_pair_length, params.batch_size):  # min_pair_length

            loss1 = model(count, shuffle_indices_nets, nodes_idx_nets, neigh_idx_nets, params.alpha, params.beta)
            loss1.backward()

            optimizer.step()  # only embedding  updated
            running_loss += loss1.detach().item()

            num_batches += 1
            if int(num_batches % 100) == 0:
                print(num_batches, " batches completed\n")
            elif not fifty and (count / min_pair_length) * 100 > 50:
                print("############# 50% epoch is completed #################\n")
                fifty = True

            optimizer.zero_grad()
            if device != 'cpu':
                torch.cuda.empty_cache()
                gc.collect()

        model.classifier_w.requires_grad = True
        model.classifier_b.requires_grad = True
        for n_net in range(params.nviews):  # len(G)
            model.attention[n_net].requires_grad = True
            model.nn_linears[n_net].weight.requires_grad = True
            model.b_att[n_net].requires_grad = True
            model.node_embeddings[n_net].weight.requires_grad = False
            model.neigh_embeddings[n_net].weight.requires_grad = False
            model.node_embeddings[n_net].weight.grad = None
            model.neigh_embeddings[n_net].weight.grad = None

        optimizer.zero_grad()

        loss3 = model.supervision_link_multi_class(labels, common_nodes, train_splits_cur, ytrain, params.gamma)
        loss3.backward()

        optimizer.step()  # update occurs except embeddings
        running_loss += loss3.detach()

        total_loss = running_loss / (num_batches)
        elapsed = time.time() - start_init
        print('epoch=', epo, '\t time=', elapsed, ' seconds\t total_loss=', total_loss)

    concat_tensors = model.node_embeddings[0].weight.detach().cpu()
    print('Embedding of view ', 1, ' ', concat_tensors)

    for i_tensor in range(1, model.num_net):
        print('Embedding of view ', (i_tensor + 1), ' ', model.node_embeddings[i_tensor].weight.detach().cpu())
        concat_tensors = torch.cat((concat_tensors, model.node_embeddings[i_tensor].weight.detach().cpu()), 1)
    #

    embed_result = np.array(concat_tensors)

    emb_file = params.output + params.dataset + "/Embedding_" + "concatenated" + str(
        params.cur_split) + '_epoch_' + str(
        epo) + "_" + ".txt"

    fo = open(emb_file, 'a+')
    for idx in range(len(embed_result)):
        word = (idx2node[idx])
        fo.write(word + ' ' + ' '.join(map(str, embed_result[idx])) + '\n')
    fo.close()


if __name__ == '__main__':
    main()
