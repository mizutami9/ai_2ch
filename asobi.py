import MeCab

m=MeCab.Tagger('-Ochasen')
print((MeCab.Tagger('-Owakati')).parse('すももももももももものうち'))