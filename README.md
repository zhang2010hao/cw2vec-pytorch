# cw2vec-pytorch

word2vec原文链接：http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
                 https://arxiv.org/pdf/1301.3781.pdf

fasttext原文链接：https://arxiv.org/pdf/1607.04606.pdf

cw2vec原文链接：http://www.statnlp.org/wp-content/uploads/papers/2018/cw2vec/cw2vec.pdf

cw2vec可以看做中文版的fasttext，原理上有一部分相似之处。这是一个pytorch实现的cw2vec，非官方版本

## 数据
所有需要用到的数据到在resource/data目录下,其中zhihu.txt是训练语料，strokes.txt是字对应的笔画，word_strokes.txt是已经做好3-5 n-gram的词，你也可以修改程序，不使用word_strokes.txt，直接用字去组成词的n-gram

## 使用方法
python cw2vec.py resource/data/zhihu.txt resource/model/ resource/data/cw2vec.vec resource/data/word_strokes.txt resource/data/strokes.txt

## Tips
1.代码中构建batch的过程与https://blog.csdn.net/mr_tyting/article/details/80091842博客类似，具体可参考此博客
2.本实现里对数据进行了subsampling，原文中没有提到
3.本实现参考https://github.com/kefirski/pytorch_NEG_loss中的损失函数
4.在表示词的时候，本实现使用笔画嵌入的均值与词嵌入的和作为词的表示，此处跟原文有不同

## 后记
还是希望原文作者可开源一下官方实现，本实现仅供参考，如有疑问可以在issues中留言探讨
