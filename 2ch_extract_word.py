def extract_word(file_name, ex_word):
    with open(file_name, 'r') as f:
        with open(file_name.split(".")[0] + "_extby_" + ex_word[0] + ".txt", "w") as g:
            for line in f:
                ins_f = lambda x: x in line
                if any(map(ins_f, ex_word)):
                    g.write(line)


if __name__ == '__main__':
    extract_word("2ch_scraped_list.txt", ["baseball", "野球"])
