import codecs
import re
# title=[]
with codecs.open('titles.txt','r',encoding='utf-8') as f:
    with open('titledata_aaa.txt', 'w') as g:
        for line in f:
            if '<a class=\"tl\"' in line:
                g.write(re.sub('<a class=\"tl\" id=\"best\d+\" target=\"_blank\" href=\"https://ncode.syosetu.com/n[0-9][0-9][0-9][0-9][a-z]+/\">', '', line)[:-6]+'\n')
                # title.append(re.sub('<a class=\"tl\" id=\"best\d+\" target=\"_blank\" href=\"https://ncode.syosetu.com/n[0-9][0-9][0-9][0-9][a-z]+/\">', '', line)[:-6])

    # f.write(title)