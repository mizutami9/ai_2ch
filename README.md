# 初めに
作成中

# 使い方
とりあえずnarou_creator.pyで学習

naroucreator2.pyで推論
## データセット
・tiledata.txt

・2ch_scraped_list.txt

・2ch_scraped_list_extby_YouTube.txt
等。

2ch_・・・が頭にあるものは2chスレタイ用データ

_extby_○○は○○で抽出したデータ

_dic.pは辞書データで、学習時に自動生成される。
そのため、推論時はこのファイルを指定してあげる必要がある。

## 学習までの流れ

2ch_extract_word.pyでワード抽出したデータセット作成

naroutitle_creator.pyで学習　学習データを上記抽出データに変更

exhaust_char_inference.pyで推論。データセットに合わせた辞書があるので、先頭行を変更し、modelを指定する
