# -*- coding:UTF-8 -*-

import numpy
import random

# numpy.random.seed(12345)

# 笔画编码映射
stroke2id = {}
for i in range(1, 6):
    for j in range(1, 6):
        for z in range(1, 6):
            stroke2id[str(i) + str(j) + str(z)] = len(stroke2id) + 1
            for m in range(1, 6):
                stroke2id[str(i) + str(j) + str(z) + str(m)] = len(stroke2id) + 1
                for n in range(1, 6):
                    stroke2id[str(i) + str(j) + str(z) + str(m) + str(n)] = len(stroke2id) + 1


class InputData:
    """
    存储数据，以及对数据进行笔画处理，生成批，进行负采样等操作
    """

    def __init__(self,
                 word2id,
                 id2word,
                 chchar2stroke,
                 file_name,
                 max_stroke,
                 n_sample,
                 word_frequency,
                 words_stroke_filename):
        self.input_file_name = file_name
        self.words_stroke_filename = words_stroke_filename
        self.max_stroke = max_stroke
        self.word2id = word2id
        self.id2word = id2word
        self.n_sample = n_sample
        self.chchar2stroke = chchar2stroke
        self.word_count = len(self.word2id)
        self.word_frequency = word_frequency
        self.stroke2id = stroke2id
        self.get_wordid_to_strokeids(words_stroke_filename, max_stroke)
        self.init_sample_table()

    def get_word_strokeids(self,
                           word,
                           smallest_n=3,
                           biggest_n=5):
        strokestr = ''
        for ch in word:
            if ch in self.chchar2stroke:
                strokestr += self.chchar2stroke[ch]

        n_gram = []
        for i in range(smallest_n, biggest_n + 1):
            j = i
            while j <= len(strokestr):
                n_gram.append(stroke2id[strokestr[j - i:j]])
                j += 1

        return n_gram

    def get_wordid_to_strokeids(self,
                                words_stroke_filename,
                                max_stroke):
        """
        创建词和笔画映射，笔画以max_stroke长度进行padding或截取，
        对于没有对应笔画的词，统一以-1对应的笔画来表示其笔画，其
        被初始化为[0,0,...,0]长度为max_stroke

        :param words_stroke_filename: 词和笔画对应文件
        :param max_stroke: 最大笔画n-gram数
        :return: 词索引和笔画索引映射
        """
        self.wordid2strokeids = {}
        with open(words_stroke_filename, 'r', encoding='utf-8-sig') as fr:
            for i in fr.readlines():
                i = i.strip().split("\t")
                if i[0] in self.word2id:
                    strokes = eval(i[2])
                    strokes_transform = [stroke2id[stroke] for stroke in strokes]
                    if max_stroke > len(strokes_transform):
                        # 默认取3,4,5的n-gram词的最长特征为363，因此对长度不够的进行padding，填充0
                        strokes_transform = strokes_transform + [0] * (max_stroke - len(strokes_transform))
                    else:
                        # 多于363的截取
                        strokes_transform = strokes_transform[:max_stroke]

                    self.wordid2strokeids[self.word2id[i[0]]] = strokes_transform

        # 当词在word2id中存在但在wordid2strokeids中不存在时统一用-1对应的笔画表示
        self.wordid2strokeids[-1] = [0] * max_stroke

        for word, id in self.word2id.items():
            if word not in self.wordid2strokeids:
                strokes_transform = self.get_word_strokeids(word)
                if max_stroke > len(strokes_transform):
                    # 默认取3,4,5的n-gram词的最长特征为363，因此对长度不够的进行padding，填充0
                    strokes_transform = strokes_transform + [0] * (max_stroke - len(strokes_transform))
                else:
                    # 多于363的截取
                    strokes_transform = strokes_transform[:max_stroke]

                self.wordid2strokeids[id] = strokes_transform

    def init_sample_table(self):
        """
        初始化负抽样
        :return:
        """

        self.sample_table = []
        sample_table_size = 1e8
        pow_frequency = numpy.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        self.ratio = pow_frequency / words_pow
        count = numpy.round(self.ratio * sample_table_size)
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
        self.sample_table = numpy.array(self.sample_table)


    def get_batch_pairs(self,
                        batch_size,
                        window_size,
                        word_data_ids,
                        shuffle=True):
        if shuffle:
            lst = list(range(len(word_data_ids)))
            random.shuffle(lst)
            word_data_ids = [word_data_ids[i] for i in lst]

        u_word_strokes = []
        v_word_strokes = []
        v_neg_strokes = []
        lens = [len(li) for li in word_data_ids]
        print("word numbers is:", sum(lens))
        for k, linearr in enumerate(word_data_ids):
            len_line = len(linearr)
            for i, u in enumerate(linearr):
                for j in range(max(0, i - window_size), min(i + window_size + 1, len_line)):
                    if len(u_word_strokes) == batch_size:
                        yield u_word_strokes, v_word_strokes, v_neg_strokes
                        u_word_strokes = []
                        v_word_strokes = []
                        v_neg_strokes = []

                    if i == j:
                        continue
                    u_word_strokes.append(self.wordid2strokeids[u])
                    v_word_strokes.append(self.wordid2strokeids[linearr[j]])
                    v_neg = self.get_neg_v_neg_sampling(self.n_sample)
                    v_neg_strokes.append([self.wordid2strokeids[neg] for neg in v_neg])

        yield u_word_strokes, v_word_strokes, v_neg_strokes


    def get_neg_v_neg_sampling(self,
                               n_sample):
        """
        根据传入的参数进行负采样

        :param n_sample: 每个样本对应的采样数
        :return:
        """
        neg_v = numpy.random.choice(
            self.sample_table, size=n_sample).tolist()
        # neg_word_strokes = [[self.wordid2strokeids[j] for j in  i] for i in neg_v]
        # neg_word_p = [self.ratio[idx] for idx in neg_v]
        # neg_v shape:[batch_size, n_neg_sample], neg_word_p shape:[batch_size, n_neg_sample]
        return neg_v

    def evaluate_pair_count(self,
                            words_num,
                            window_size,
                            data_len):
        return words_num * (2 * window_size - 1) - (
                data_len - 1) * (1 + window_size) * window_size
