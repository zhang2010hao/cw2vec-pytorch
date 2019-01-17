# -*- coding:UTF-8 -*-

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SkipGramModel(nn.Module):

    def __init__(self,
                 stroke_size,
                 word_size,
                 emb_dimension,
                 use_cuda=False):
        """
        初始化模型参数

        Args:
            stroke_size: 笔画n-gram的特征数
            emb_dimention: 笔画特征维度

        Returns:
            None
        """
        super(SkipGramModel, self).__init__()
        self.emb_dimension = emb_dimension
        self.stroke_size = stroke_size
        self.word_size = word_size
        self.use_cuda = use_cuda
        self.u_embeddings = nn.Embedding(stroke_size, emb_dimension)
        self.v_embeddings = nn.Embedding(word_size, emb_dimension)
        if use_cuda:
            self.u_embeddings.cuda()
            self.v_embeddings.cuda()

        self.init_emb()

    def init_emb(self):
        """
        初始化笔画嵌入
        """
        initrange = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self,
                max_stroke_len,
                n_sample,
                pos_u,
                pos_v,
                neg_v):
        """
        使用相似度计算损失函数

        Args:
            max_stroke_len：一个词的最大n-gram数
            n_neg_sample：负抽样个数
            pos_u: 正样本
            pos_v: 上下文
            neg_v: 负抽样数据

        Returns:
            Loss of this process, a pytorch variable.
        """
        emb_u = self.u_embeddings(pos_u)
        emb_u = torch.mean(emb_u, dim=1)
        emb_v = self.v_embeddings(pos_v)
        emb_v = torch.mean(emb_v, dim=1)
        log_target = (emb_u * emb_v).sum(1).squeeze().sigmoid().log()

        neg_emb_v = self.v_embeddings(neg_v.view(-1, max_stroke_len))
        neg_emb_v = torch.mean(neg_emb_v, dim=1).view(-1, n_sample, self.emb_dimension).neg()
        sum_log_sampled = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).sigmoid().log().sum(1).squeeze()

        loss = log_target + sum_log_sampled

        return -loss.mean()

    def save_embedding(self,
                       id2word,
                       wordid2strokeids,
                       file_name,
                       use_cuda):
        """
        保存词嵌入
        """
        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        stroke_vec = np.array([value for key, value in wordid2strokeids.items()])
        stroke_vec = Variable(torch.LongTensor(stroke_vec))
        if use_cuda:
            stroke_vec = stroke_vec.cuda()

        emb = self.u_embeddings(stroke_vec)
        emb = torch.mean(emb, dim=1)
        norm_emb = emb / torch.sqrt(torch.sum(torch.pow(emb, 2), 1)).view(emb.size(0), -1)
        if use_cuda:
            norm_emb = norm_emb.data.cpu().numpy()
        else:
            norm_emb = norm_emb.data.numpy()

        i = 0
        for wid, strokes in wordid2strokeids.items():
            if wid in id2word:
                word = id2word[wid]
                emb = norm_emb[i]
                outstr = ' '.join(map(lambda x: str(x), emb))
                fout.write('%s %s\n' % (word, outstr))
            i += 1
