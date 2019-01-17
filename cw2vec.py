# -*- coding:UTF-8 -*-

from data import InputData
from skip_gram import SkipGramModel
from data_processing import load_data, load_strokes
from torch.autograd import Variable
import torch
import torch.optim as optim
from tqdm import tqdm
import sys
import os
import time


def load_model(path,
               model):
    """
    加载模型
    :param path:
    :param model:
    :return:
    """
    if os.path.exists(path + 'model.pkl'):
        model.load_state_dict(torch.load(os.path.join(path, 'model.pkl')))
    return model


def save_model(path,
               model):
    """
    保存模型
    :param path:
    :param model:
    :return:
    """

    torch.save(model.state_dict(), os.path.join(path, 'model.pkl'))


class CW2Vec:
    def __init__(self,
                 input_file_name,
                 model_file,
                 output_file_name,
                 words_stroke_filename,
                 stroke_path,
                 emb_dimension=100,
                 batch_size=500,
                 window_size=5,
                 iteration=1,
                 initial_lr=0.025,
                 min_count=5,
                 stroke_size=3876,
                 stroke_max_length=363,
                 n_neg_sample=5):
        """
        初始化模型参数


        Returns:
            None.
        """
        self.stroke_path = stroke_path
        self.input_file_name = input_file_name
        self.min_count = min_count
        self.model_file = model_file
        self.words_stroke_filename = words_stroke_filename
        self.output_file_name = output_file_name
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.window_size = window_size
        self.iteration = iteration
        self.initial_lr = initial_lr
        self.n_neg_sample = n_neg_sample
        self.stroke_size = stroke_size
        self.use_cuda = torch.cuda.is_available()
        self.stroke_max_length = stroke_max_length
        # 加载并处理数据
        self.words_num, self.word_data_ids, self.word_frequency, self.word2id, self.id2word = load_data(
            self.input_file_name,
            self.min_count)
        # 实例化skip-gram模型
        self.skip_gram_model = SkipGramModel(stroke_size,
                                             len(self.word2id),
                                             emb_dimension,
                                             self.use_cuda)
        if os.path.exists(model_file + 'model.pkl'):
            print("loading trained model at:", model_file + 'model.pkl')
            self.skip_gram_model = load_model(model_file, self.skip_gram_model)
        self.optimizer = optim.SGD(
            self.skip_gram_model.parameters(), lr=self.initial_lr)

    def train(self,
              epochs=3):

        # 加载数据并进行相关转换

        print('Word Count: %d' % len(self.word2id))
        print('All word num: %d' % (self.words_num))

        chchar2stroke = load_strokes(self.stroke_path)
        # 初始化数据处理，生成batch
        data = InputData(self.word2id,
                         self.id2word,
                         chchar2stroke,
                         self.input_file_name,
                         self.stroke_max_length,
                         self.n_neg_sample,
                         self.word_frequency,
                         self.words_stroke_filename)

        batch_count = 2 * self.window_size * (self.words_num - 1) // self.batch_size + 1


        for epoch in range(1, epochs + 1):
            # 初始化进度条
            process_bar = tqdm(total=batch_count)
            dataiter = data.get_batch_pairs(self.batch_size, self.window_size, self.word_data_ids)

            i = 0
            for u_word_strokes, v_word_strokes, v_neg_strokes in dataiter:
                i += 1
                pos_u = Variable(torch.LongTensor(u_word_strokes))
                pos_v = Variable(torch.LongTensor(v_word_strokes))
                neg_v = Variable(torch.LongTensor(v_neg_strokes))

                if self.use_cuda:
                    pos_u = pos_u.cuda()
                    pos_v = pos_v.cuda()
                    neg_v = neg_v.cuda()

                self.optimizer.zero_grad()
                loss = self.skip_gram_model.forward(self.stroke_max_length,
                                                    self.n_neg_sample,
                                                    pos_u,
                                                    pos_v,
                                                    neg_v)

                loss.backward()
                self.optimizer.step()

                process_bar.set_description("Epoch: %d, Iter_num: %d, Loss: %0.8f, lr: %0.6f" %
                                            (epoch,
                                             i * self.batch_size,
                                             loss.data[0],
                                             self.optimizer.param_groups[0]['lr']))

                process_bar.update(1)

                if i * self.batch_size % 200000 == 0:
                    lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr

            print()
            print("epoch %d finished, save embedding" % (epoch))
            self.skip_gram_model.save_embedding(self.id2word,
                                                data.wordid2strokeids,
                                                self.output_file_name,
                                                self.use_cuda)
            print('saver the new model')
            save_model(self.model_file, self.skip_gram_model)
            process_bar.close()


if __name__ == '__main__':
    w2v = CW2Vec(input_file_name=sys.argv[1],
                 model_file=sys.argv[2],
                 output_file_name=sys.argv[3],
                 words_stroke_filename=sys.argv[4],
                 stroke_path=sys.argv[5])
    w2v.train()
