<<<<<<< HEAD
# 過去ログからF12で拾ってこい
# https://eagle.5ch.net/livejupiter/kako/kako0000.html
#こういうところから、f12して kako○○.htmlファイルをdlして2ch_threadフォルダに格納して、これ自体を実行するとデータが出来上がる

import glob
import  codecs
import re

files = glob.glob("./2ch_thread/*")
with open('2ch_scraped_list.txt', 'w') as g:
    for file in files:
        print(file)
        with codecs.open(file, 'r', encoding='utf-8') as f:
            list_flag=0
            for line in f:
                if list_flag == 1 and '<hr>' in line:
                    list_flag = 0
                    print('decfcccccccc')
                if list_flag == 1:
                    tmp = re.sub('<p class=\"main_[a-z]*\">[0-9]+<span class=\"filename\">'
                                 '[0-9]+.dat</span><span class=\"title\"><a href=\"/test/read.cgi/news4vip'
                                 '/[0-9]+/\">','',line)
                    tmp = re.sub('</a></span><span class=\"lines\">[0-9]+</span></p>','',tmp)
                    print(tmp)
                    g.write(tmp)
                if list_flag == 2:
                    list_flag = 1
                if 'THREAD List' in line:
                    list_flag = 2
=======

# 過去ログからF12で拾ってこい
# https://eagle.5ch.net/livejupiter/kako/kako0000.html
#こういうところから、f12して kako○○.htmlファイルをdlして2ch_threadフォルダに格納して、これ自体を実行するとデータが出来上がる

import glob
import  codecs
import re

files = glob.glob("./2ch_thread/*")

for file in files:
    print(file)
    with codecs.open(file, 'r', encoding='utf-8') as f:
        with open('2ch_scraped_list.txt', 'w') as g:
            list_flag=0
            for line in f:
                if list_flag == 1 and '<hr>' in line:
                    list_flag = 0
                    print('decfcccccccc')
                if list_flag == 1:
                    tmp = re.sub('<p class=\"main_[a-z]*\">[0-9]+<span class=\"filename\">'
                                 '[0-9]+.dat</span><span class=\"title\"><a href=\"/test/read.cgi/news4vip'
                                 '/[0-9]+/\">','',line)
                    tmp = re.sub('</a></span><span class=\"lines\">[0-9]+</span></p>','',tmp)
                    print(tmp)
                    g.write(tmp)
                if list_flag == 2:
                    list_flag = 1
                if 'THREAD List' in line:
                    list_flag = 2
>>>>>>> 69fc9c8e2e4460b587cbc3505d375c6c48a4ae6e
