import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import random
import pickle
import cloudpickle


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


def train(model, criterion, input, target):
    hidden = model.initHidden()

    model.zero_grad()

    output, _ = model(input, hidden)
    _, predY = torch.max(output.data, 1)
    loss = criterion(output, target)

    loss.backward()

    return loss.data.item() / input.size()[0]


def inputTensor(input_idx):
    tensor = torch.LongTensor(input_idx)
    return autograd.Variable(tensor)


def targetTensor(input_idx, char_idx):
    input_idx = input_idx[1:]
    input_idx.append(char_idx['EOS'])
    tensor = torch.LongTensor(input_idx)
    return autograd.Variable(tensor)


def train_main(train_data, e_dim=256, h_dim=256):
    titles = []
    with open(train_data, 'r') as f:
        for i in f:
            titles.append(i)
    # print(titles)

    all_char_str = set([char for name in titles for char in name])

    char2idx = {char: i for i, char in enumerate(all_char_str)}
    char2idx['EOS'] = len(char2idx)

    # save dictionary
    pickle.dump(char2idx, open(train_data + "_dic.p", "wb"))

    names_idx = [[char2idx[char_str] for char_str in name_str] for name_str in titles]
    # print(names_idx)

    # build model
    model = LSTM(input_dim=len(char2idx), embed_dim=e_dim, hidden_dim=h_dim)
    criterion = nn.NLLLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    n_iters = 1000
    all_losses = []

    best_loss = 1000000
    no_improve_iter = 0

    for iter in range(1, n_iters + 1):

        # data shuffle
        random.shuffle(names_idx)

        total_loss = 0

        for i, name_idx in enumerate(names_idx):
            inputa = inputTensor(name_idx)
            target = targetTensor(name_idx, char2idx)

            loss = train(model, criterion, inputa, target)
            total_loss += loss

            optimizer.step()

        print(iter, "/", n_iters)
        print("loss {:.4}".format(float(total_loss / len(names_idx))))
        if best_loss > float(total_loss / len(names_idx)):
            best_loss = float(total_loss / len(names_idx))
            no_improve_iter = 0
            torch.save(model.state_dict(), "./models/" + train_data + "_{:06}.model".format(iter))
        else:
            no_improve_iter += 1
            if no_improve_iter > 30:
                print('early stop!')
                break

    # with open('./model/model_narou_char_1000.pkl', 'wb')as f:
    #     cloudpickle.dump(model, f)


if __name__ == '__main__':
    # train_main("titledata.txt", e_dim=256, h_dim=256)
    # train_main("2ch_scraped_list_extby_App.txt", e_dim=256, h_dim=256)

    train_list = ["2ch_scraped_list_extby_kyuubo.txt",
                  "2ch_scraped_list_extby_moneeeyy.txt", "2ch_scraped_list_extby_nanj.txt",
                  "2ch_scraped_list_extby_old.txt","2ch_scraped_list_extby_sosyage.txt",
                                                   "2ch_scraped_list_extby_baseball.txt"
                  ]
    for i in train_list:
        train_main(i, e_dim=256, h_dim=256)