"""
by Qiu Lisong before 2019/10
"""


import sys

MAX_GRAM = 4

def make_n_gram(words, n):
    lenw = len(words)
    words = ["BBBB"] * (n - 1) + words + ["EEEE"] * (n - 1)
    ngrams = []

    for i in range(lenw):
        str = ""
        for j in range(n):
            str += words[i + j] + "_"
        str = str[:-1]
        ngrams.append(str)
        i += n
    return ngrams


def make_all_n_gram(datas):
    all_words = len(datas)
    all_ngram = [[] for i in range(4)]
    all_ngram_len = []
    all_ngram_score = []

    all_ngram[0] += make_n_gram(datas, 1)
    all_ngram[1] += make_n_gram(datas, 2)
    all_ngram[2] += make_n_gram(datas, 3)
    all_ngram[3] += make_n_gram(datas, 4)

    for i in range(4):
        all_ngram_len.append(get_uniq_length(all_ngram[i]))
        all_ngram_score.append(all_ngram_len[i] * 1.0 / max(1.0, all_words-i))

    return all_ngram, all_ngram_len, all_words, all_ngram_score


def get_uniq_length(ngram):
    return len(list(set(ngram)))


def uniq_n_gram_length(words, n):
    ngrams = make_n_gram(words, n)
    return len(list(set(ngrams)))


def distinct(words):
    dis = []

    word_count = len(words)
    if 0 == word_count:
        return [], 0, []
    dis_score = []

    for diss in dis:
        dis_score.append(diss * 1.0 / word_count)

    return dis, word_count, dis_score


def readinput():
    datas = []
    for line in sys.stdin:
        words = line.strip().split()
        datas.append(words)
    return datas


class Inter_Distinct:
    def __init__(self):
        return

    def compute_score(self, gts, res):
        datas = []
        for id in range(len(res)):
            lines = res[id]
            for line in lines:
                words = line.split()
                if len(words) == 0:
                    continue
                datas.append(words)

        dis_scores = deal_by_all(datas)
        score = [dis_scores['dist-1'], dis_scores['dist-2'], dis_scores['dist-3'], dis_scores['dist-4']]

        return score, dis_scores

    def method(self):
        return "Inter_Distinct"


def deal_by_all(datas):
    dataall = []
    for data in datas:
        dataall += data

    try:
        ngram, dis, word_count, dis_score = make_all_n_gram(dataall)
    except Exception:
        dis_score = [0, 0, 0, 0]

    dis_scores = {'dist-1': dis_score[0],
                  'dist-2': dis_score[1],
                  'dist-3': dis_score[2],
                  'dist-4': dis_score[3],}

    return dis_scores