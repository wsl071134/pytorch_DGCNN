from dgcnn_data_util import load_data, get_regex_dict, get_tag_dict, get_embeddings_dict
import torch
import numpy as np
from train_dgcnn import Classifier


def test():
    regex_dict = get_regex_dict()
    tag_dict = get_tag_dict()
    embedding_matrix, stmt_dict = get_embeddings_dict()
    train_graphs, test_graphs = load_data(regex_dict, tag_dict, embedding_matrix, stmt_dict)
    g_list = train_graphs
    g_list.extend(test_graphs)
    lable = []
    for g in g_list:
        lable.append(g.label)
    net = torch.load('model/model.pt')
    net.eval()
    pre = net(g_list)
    print(pre)
    print(lable)
    print(type(pre[1]))
    p = pre[0].detach().numpy()
    indices = [np.argmax(x) for x in p]
    print(indices)
    sum=0
    for i, j in zip(lable, indices):
        sum += i == j
    print('{:.2%}'.format(sum/len(lable)))

if __name__ == '__main__':
    test()
