import pickle

char2idx = pickle.load(open("dic.p", "rb"))
idx2char = {v: k for k, v in char2idx.items()}

print(idx2char[87])