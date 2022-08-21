import naroucreator2 as nc

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import random
import pickle
import cloudpickle
import pprint

from collections import defaultdict

if __name__ == '__main__':
    # load dic
    # char2idx = pickle.load(open("titledata.txt_dic.p", "rb"))
    char2idx = pickle.load(open("2ch_scraped_list_extby_YouTube.txt" + "_dic.p", "rb"))
    idx2char = {v: k for k, v in char2idx.items()}
    # build model
    model = nc.LSTM(input_dim=len(char2idx), embed_dim=256, hidden_dim=256)
    model.eval()
    # with open('./models/model_narou_char_max_iter.pkl', 'rb') as f:
    #     # model = cloudpickle.load(f)
    # model.load_state_dict(torch.load('./models/titledata.txt' + "_000037.model"))
    model.load_state_dict(torch.load('./models/2ch_scraped_list_extby_YouTube.txt' + "_000002.model"))

    counter_i = defaultdict(lambda: 0)
    counter = defaultdict(lambda: 0)
    with open("2ch_scraped_list_extby_YouTube.txt", "r") as f:
        for line in f:
            counter_i[line[0]] += 1
            for l in line:
                counter[l] += 1

    # print(sorted(counter_i.items(), key=lambda x: x[1], reverse=True))
    # print(sorted(counter.items(), key=lambda x: x[1], reverse=True))

    sorted_c_i = sorted(counter_i.items(), key=lambda x: x[1], reverse=True)

    # print(len(counter_i), len(counter))

    print(sorted_c_i)

    init_list = [i[0] for i in sorted_c_i]

    print(init_list)

    nc.samples(char2idx,model,init_list[0:5])
