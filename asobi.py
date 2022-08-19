# import MeCab
#
# m=MeCab.Tagger('-Ochasen')
# print((MeCab.Tagger('-Owakati')).parse('すももももももももものうち'))

import requests

i = 1

site_data = requests.get("https://eagle.5ch.net/livejupiter/kako/kako0500.html".format(i))
# site_data = requests.get("https://eagle.5ch.net/livejupiter/kako/kako{:04}.html".format(i))
site_data.encoding = site_data.apparent_encoding

with open("2ch_thread/a.html", "w", encoding='utf-8') as f:
    f.write(site_data.text)
