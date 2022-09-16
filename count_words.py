import MeCab
from collections import defaultdict


m=MeCab.Tagger('-Ochasen')

mecabed = MeCab.Tagger('-Owakati').parse('すももももももももものうち')
# print(type(mecabed.split()))

train_data = "2ch_scraped_list_extby_Anime.txt"

counter_i = defaultdict(lambda: 0)
counter = defaultdict(lambda: 0)
with open(train_data, "r") as f:
    for line in f:
        mecabed = MeCab.Tagger('-Owakati').parse(line)
        counter_i[line[0]] += 1
        for l in mecabed.split():
            counter[l] += 1

# print(sorted(counter_i.items(), key=lambda x: x[1], reverse=True))
# print(sorted(counter.items(), key=lambda x: x[1], reverse=True))

sorted_c_i = sorted(counter.items(), key=lambda x: x[1], reverse=True)

print(sorted_c_i)