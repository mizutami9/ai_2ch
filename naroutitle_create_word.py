from text_encoder import JapaneseTextEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import pickle
import cloudpickle


class RNNLM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, batch_size=50, num_layers=1):
        super(RNNLM, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=0.5)

        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, num_layers=self.num_layers)

        self.output = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self):
        self.hidden_state = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)

    def forward(self, indices):
        embed = self.word_embeddings(indices)  # batch_len x sequence_length x embedding_dim
        drop_out = self.dropout(embed)
        if drop_out.dim() == 2:
            drop_out = torch.unsqueeze(drop_out, 1)
        gru_out, self.hidden_state = self.gru(drop_out, self.hidden_state)  # batch_len x sequence_length x hidden_dim
        gru_out = gru_out.contiguous()
        return self.output(gru_out)


def train2batch(dataset, batch_size):
    batch_dataset = []
    for i in range(0, 6, batch_size):
        batch_dataset.append(dataset[i:i + batch_size])
    return batch_dataset


def training_main():
    titles = []
    with open('titledata.txt', 'r')as f:
        for i in f:
            titles.append(i)

    encoder = JapaneseTextEncoder(titles, append_eos=True, maxlen=100, padding=True)
    encoder.build()
    # with open('word_data.pkl', 'wb')as f:
    #     cloudpickle.dump(encoder, f)

    indices = encoder.dataset[0]
    print(indices)
    print(encoder.decode(indices))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 50
    n_vocab = len(encoder.word2id)
    n_epoch = 100000
    EMBEDDING_DIM = HIDDEN_DIM = 256
    model = RNNLM(EMBEDDING_DIM, HIDDEN_DIM, n_vocab).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Training
    model.train()
    min_loss = 1000
    no_improve_iter = 0
    for epoch in range(1, n_epoch + 1):
        epoch_loss = 0

        encoder.shuffle()
        # len(encoder.dataset) == 1361
        batch_dataset = train2batch(encoder.dataset, batch_size)
        for batch_data in batch_dataset:
            model.zero_grad()
            model.init_hidden()

            batch_tensor = torch.tensor(batch_data, device=device)
            # print(batch_tensor.size())
            input_tensor = batch_tensor[:, :-1]
            target_tensor = batch_tensor[:, 1:].contiguous()
            outputs = model(input_tensor)
            outputs = outputs.view(-1, n_vocab)
            targets = target_tensor.view(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            epoch_loss += loss.data.item() / input_tensor.size()[0]
            optimizer.step()
        print(epoch, ' / ', n_epoch,'loss:',epoch_loss/len(batch_dataset))
        if min_loss > epoch_loss/len(batch_dataset):
            min_loss=epoch_loss/len(batch_dataset)
            no_improve_iter=0
            torch.save(model.state_dict(), "./models/word_narou_word_{:06}.model".format(epoch))
        else:
            no_improve_iter+=1
            if no_improve_iter>19:
                print('early stop!')
                break

    torch.save(model.state_dict(), "./models/word_narou_word_max_iter.model")


def test_main():
    # with open('word_data.pkl', 'rb')as f:
    #     encoder = cloudpickle.dump(f)
    titles = []

    with open('titledata.txt', 'r')as f:
        for i in f:
            titles.append(i)
    encoder = JapaneseTextEncoder(titles, append_eos=True, maxlen=50, padding=True)
    encoder.build()
    print(encoder.dataset)
    EMBEDDING_DIM = HIDDEN_DIM = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_vocab = len(encoder.word2id)
    model = RNNLM(EMBEDDING_DIM, HIDDEN_DIM, n_vocab, batch_size=1).to(device)
    model_name = "./models/word_narou_word_max_iter.model"
    model.load_state_dict(torch.load(model_name))

    # Evaluation
    model.eval()
    with torch.no_grad():
        for i in range(30):
            model.init_hidden()
            morpheme = "チート"
            sentence = [morpheme]
            for j in range(50):
                index = encoder.word2id[morpheme]
                input_tensor = torch.tensor([index], device=device)
                outputs = model(input_tensor)
                probs = F.softmax(torch.squeeze(outputs))
                p = probs.cpu().detach().numpy()
                morpheme = np.random.choice(encoder.vocab, p=p)
                sentence.append(morpheme)
                if morpheme in ["</s>", "<pad>"]:
                    break
            print("".join(sentence))



if __name__ == '__main__':
    # training_main()
    test_main()
