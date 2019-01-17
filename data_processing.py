# -*- coding:UTF-8 -*-

import numpy
from collections import Counter

numpy.random.seed(12345)


def load_data(input_file_name, min_count, padding='<PAD>'):
    """
    加载数据，并生成数据集，词典，词频等
    :param input_file_name:
    :param min_count:
    :param padding:
    :return:
    """
    counter = Counter()
    words_num = 0
    word_data_ids = []
    word_data_tmp = []
    with open(input_file_name, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            line = line.strip().split(' ')
            words_num += len(line)
            word_data_tmp.append(line)
            counter.update(line)

    word2id = dict()
    id2word = dict()
    wid = 0
    word_frequency = dict()
    for w, c in counter.items():
        if c < min_count:
            words_num -= c
            continue
        word2id[w] = wid
        id2word[wid] = w
        word_frequency[wid] = c
        wid += 1
    word2id[padding] = wid
    id2word[wid] = padding

    for linearr in word_data_tmp:
        linearr_tmp = [word2id[word] for word in linearr if word in word2id]
        if len(linearr_tmp) > 0:
            word_data_ids.append(linearr_tmp)

    return words_num, word_data_ids, word_frequency, word2id, id2word


def load_strokes(stroke_path):
    """
    加载字对应的笔画，并转为论文中对应的编码
    :param stroke_path:
    :return:
    """
    stroke2id = {'横': '1', '提': '1', '竖': '2', '竖钩': '2', '撇': '3', '捺': '4', '点': '4'}
    chchar2stroke = {}

    with open(stroke_path, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            line = line.strip().split(':')
            if len(line) == 2:
                arr = line[1].split(',')
                strokes = [stroke2id[stroke] if stroke in stroke2id else '5' for stroke in arr]
                chchar2stroke[line[0]] = ''.join(strokes)

    return chchar2stroke
