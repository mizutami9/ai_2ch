
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import random
import pickle
import cloudpickle
import pprint


class LSTM(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeds = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
        self.hidden = self.initHidden()

    def forward(self, input, hidden):
        embeds = self.embeds(input)
        lstm_out, hidden = self.lstm(
            embeds.view(len(input), 1, -1), hidden)
        output = self.linear(lstm_out.view(len(input), -1))
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))


def inputTensor(input_idx):
    tensor = torch.LongTensor(input_idx)
    return autograd.Variable(tensor)


def sample_mn(char2idx,model,start_letter='ア', max_length=100):
    sample_char_idx = [char2idx[start_letter]]
    idx2char = {v: k for k, v in char2idx.items()}

    input = inputTensor(sample_char_idx)

    hidden = model.initHidden()

    m = 3
    n = 3

    output_name = start_letter
    output_name = start_letter


    hidden = model.initHidden()

    output, hidden = model(input, hidden)
    print(idx2char[int(output.data.topk(1)[1][0][0])])
    print(idx2char[int(output.data.topk(2)[1][0][1])])
    print(idx2char[int(output.data.topk(3)[1][0][2])])
    print((output.data.topk(4)))


    all_output = []
    for j in range(m):
        for k in range(n):
            output_name = start_letter

            input = inputTensor(sample_char_idx)

            hidden = model.initHidden()

            output, hidden = model(input, hidden)
            _, topi = output.data.topk(j + 1)
            topi = topi[0][j]
            if topi == char2idx['EOS']:
                break
            else:
                letter = idx2char[int(topi)]
                output_name += letter
            input = inputTensor([topi])
            output, hidden = model(input, hidden)
            _, topi = output.data.topk(k + 1)
            topi = topi[0][k]
            if topi == char2idx['EOS']:
                break
            else:
                letter = idx2char[int(topi)]
                output_name += letter
            input = inputTensor([topi])
            for i in range(max_length):
                output, hidden = model(input, hidden)
                _, topi = output.data.topk(1)
                topi = topi[0][0]
                if topi == char2idx['EOS'] or topi == char2idx['\n']:
                    break
                else:
                    letter = idx2char[int(topi)]
                    output_name += letter
                input = inputTensor([topi])
            all_output.append(output_name)

    return all_output


def sample(char2idx,model, start_letter='ア', max_length=100):
    sample_char_idx = [char2idx[start_letter]]
    idx2char = {v: k for k, v in char2idx.items()}

    input = inputTensor(sample_char_idx)

    hidden = model.initHidden()

    output_name = start_letter

    for i in range(max_length):
        output, hidden = model(input, hidden)
        _, topi = output.data.topk(1)
        topi = topi[0][0]
        if topi == char2idx['EOS']:
            break
        else:
            letter = idx2char[int(topi)]
            output_name += letter
        input = inputTensor([topi])

    return output_name


def samples(char2idx,model,start_letters='アイウ'):
    for start_letter in start_letters:
        pprint.pprint(sample_mn(char2idx,model,start_letter))


def main(char2idx,model):
    samples(char2idx,model,u'あい')


if __name__ == '__main__':
    # load dic
    # char2idx = pickle.load(open("titledata.txt_dic.p", "rb"))
    char2idx = pickle.load(open("2ch_scraped_list_extby_YouTube.txt" + "_dic.p", "rb"))
    idx2char = {v: k for k, v in char2idx.items()}
    # build model
    model = LSTM(input_dim=len(char2idx), embed_dim=256, hidden_dim=256)
    model.eval()
    # with open('./models/model_narou_char_max_iter.pkl', 'rb') as f:
    #     # model = cloudpickle.load(f)
    # model.load_state_dict(torch.load('./models/titledata.txt' + "_000037.model"))
    model.load_state_dict(torch.load('./models/2ch_scraped_list_extby_YouTube.txt' + "_000002.model"))

    main(char2idx,model)

