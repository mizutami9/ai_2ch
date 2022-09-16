import MeCab

m=MeCab.Tagger('-Ochasen')
mecabed = MeCab.Tagger('-Owakati').parse('すももももももももものうち')
print(type(mecabed.split()))

# import requests
#
# i = 1
#
# site_data = requests.get("https://eagle.5ch.net/livejupiter/kako/kako0500.html".format(i))
# # site_data = requests.get("https://eagle.5ch.net/livejupiter/kako/kako{:04}.html".format(i))
# site_data.encoding = site_data.apparent_encoding
#
# with open("2ch_thread/a.html", "w", encoding='utf-8') as f:
#     f.write(site_data.text)

# s = "youtuber"
# t = ["hoge", "fuga", "youtubee"]
# ins_f = lambda x:x in s
# if any(map(ins_f, t)):
#     print("fizz!")

# def targetTensor(input_idx, char_idx):
#     input_idx = input_idx[1:]
#     input_idx.append(char_idx['EOS'])
#     # tensor = torch.LongTensor(input_idx)
#     return input_idx
#
#
# char_idx = {1: 1, 2: 2, 'EOS': 3}
# tem = [[1,2,4],[2,3,4],[3,4,5]]
#
# res = [targetTensor(i,char_idx) for i in tem]
#
# print(res)
#
# res=list(range(0, 100-49,49))
# res1=list(range(0, 100))
#
# print(list(range(0, 100-49,49)),res1[100//49*49:])