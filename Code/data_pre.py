# 数据准备

import torch
import numpy as np
from torch.autograd import Variable
from nltk import word_tokenize
from collections import Counter

from utils import subsequent_mask, seq_padding, get_word_dict
from setting import BATCH_SIZE, UNK, PAD, DEVICE, Train_en, Train_cn


# 数据准备：
class PrepareData:
    # 输入训练集、测试集的中英文文件路径：
    def __init__(self, file_en, file_cn, read_num=10000000, sort=True):   # read_num要读取数据的行数, sort=True对句子按长度进行排序。
        # 读取数据 并分词
        self.data_en, self.data_cn = self.load_data(file_en, file_cn, read_num)

        # 读取字典文件的信息：
        self.cn_index_dict, self.cn_word_dict, self.en_index_dict, self.en_word_dict = get_word_dict()
        # 词典的个数：
        self.En_vocab = len(self.en_index_dict)
        self.Cn_vocab = len(self.cn_index_dict)

        # 将word转成ID
        self.data_en, self.data_cn = self.wordToID(self.data_en, self.data_cn, self.en_word_dict, self.cn_word_dict, sort)


        # 划分batch + padding + mask
        self.data = self.splitBatch(self.data_en, self.data_cn, BATCH_SIZE)

    # 数据加载：中文与英文是不一样的。
    def load_data(self, path_en, path_cn, read_num):
        """
        读取翻译前(英文)和翻译后(中文)的数据文件
        每条数据都进行分词，然后构建成包含起始符(BOS)和终止符(EOS)的单词(中文为字符)列表
        形式如：en = [['BOS', 'i', 'love', 'you', 'EOS'], ['BOS', 'me', 'too', 'EOS'], ...]
                cn = [['BOS', '我', '爱', '你', 'EOS'], ['BOS', '我', '也', '是', 'EOS'], ...]
        """
        en = []
        cn = []
        # （1）英文：
        en_num = read_num
        with open(path_en, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()  # 去除字符串两边的空格（包含\n,\r,\t）
                en.append(["BOS"] + word_tokenize(line) + ["EOS"])  # 区分大小写
                en_num -= 1
                if en_num <= 0:
                    break

        #（2）中文：
        cn_num = read_num
        with open(path_cn, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                cn.append(["BOS"] + word_tokenize(" ".join([w for w in line])) + ["EOS"])
                        # " ".join([w for w in line]) 是为了在汉字之间添加空格，使得可以被word_tokenize识别！
                cn_num -= 1
                if cn_num <= 0:
                    break

        return en, cn

    # 对单词进行编码，并按英文句子的长度进行排序：
    def wordToID(self, en, cn, en_dict, cn_dict, sort=True):
        """
        该方法可以将翻译前(英文)数据和翻译后(中文)数据的单词列表表示的数据
        均转为id列表表示的数据
        如果sort参数设置为True，则会以翻译前(英文)的句子(单词数)长度排序
        以便后续分batch做padding时，同批次各句子需要padding的长度相近减少padding量
        """

        # 将翻译前(英文)数据和翻译后(中文)数据都转换为id表示的形式
        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(w, 0) for w in sent] for sent in cn]

        # 构建一个按照句子长度排序的函数
        def len_argsort(seq):
            """
            传入一系列句子数据(分好词的列表形式)，
            按照句子长度排序后，返回排序后原来各句子在数据中的索引下标
            """
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        # 把中文和英文按照同样的顺序排序
        if sort:
            # 以英文句子长度排序的(句子下标)顺序为基准
            sorted_index = len_argsort(out_en_ids)
            # 对翻译前(英文)数据和翻译后(中文)数据都按此基准进行排序
            out_en_ids = [out_en_ids[i] for i in sorted_index]
            out_cn_ids = [out_cn_ids[i] for i in sorted_index]

        return out_en_ids, out_cn_ids

    # 数据批次划分：
    def splitBatch(self, en, cn, batch_size, shuffle=True):
        """
        将以单词id列表表示的翻译前(英文)数据和翻译后(中文)数据
        按照指定的batch_size进行划分
        如果shuffle参数为True，则会对这些batch数据顺序进行随机打乱
        排序之后，一个batch深入，填充的位置会变少
        """
        # 在按数据长度生成的各条数据下标列表[0, 1, ..., len(en)-1]中
        # 每隔指定长度(batch_size)取一个下标作为后续生成batch的起始下标
        idx_list = np.arange(0, len(en), batch_size)
        # 如果shuffle参数为True，则将这些各batch起始下标打乱
        if shuffle:
            np.random.shuffle(idx_list)
        # 存放各个batch批次的句子数据索引下标
        batch_indexs = []
        for idx in idx_list:
            # 注意，起始下标最大的那个batch可能会超出数据大小
            # 因此要限定其终止下标不能超过数据大小
            """
            形如[array([4, 5, 6, 7]), 
                 array([0, 1, 2, 3]), 
                 array([8, 9, 10, 11]),
                 ...]
            """
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))

        # 按各batch批次的句子数据索引下标，构建实际的单词id列表表示的各batch句子数据
        batches = []
        for batch_index in batch_indexs:
            # 按当前batch的各句子下标(数组批量索引)提取对应的单词id列表句子表示数据
            batch_en = [en[index] for index in batch_index]
            batch_cn = [cn[index] for index in batch_index]
            # 对当前batch的各个句子都进行padding对齐长度
            # 维度为：batch数量×batch_size×每个batch最大句子长度
            batch_cn = seq_padding(batch_cn)   # seq_padding是导入的自定义函数
            batch_en = seq_padding(batch_en)
            # 将当前batch的英文和中文数据添加到存放所有batch数据的列表中
            batches.append(Batch(batch_en, batch_cn))  # 这里引用了下面的Batch类

        return batches


class Batch:
    # Object for holding a batch of data with mask during training.
    def __init__(self, src, trg=None, pad=0):
        # 将输入与输出的单词id表示的数据规范成整数类型
        src = torch.from_numpy(src).to(DEVICE).long()
        trg = torch.from_numpy(trg).to(DEVICE).long()
        self.src = src
        # 对于当前输入的句子非空部分进行判断成bool序列
        # 并在seq_length前面增加一维，形成维度为 batch_size×1×seq_length 的矩阵
        # src_mask值为0的位置是False
        self.src_mask = (src != pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对decoder要使用到的target句子进行mask
        if trg is not None:
            # decoder要用到的target输入部分
            self.trg = trg[:, :-1]   # 不包含最后一个元素
            # decoder训练时应预测输出的target结果
            self.trg_y = trg[:, 1:]  # 不包含第一个元素
            # 将target输入部分进行attention mask
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # 将应输出的target结果中实际的词数进行统计
            self.ntokens = (self.trg_y != pad).data.sum()

    # Mask掩码操作
    @staticmethod
    # 将外部函数集成到类体中，美化代码结构，在不需要类实例化的情况下调用方法。
    def make_std_mask(tgt, pad):
        # Create a mask to hide padding and future words.
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


if __name__ == '__main__':
    data_test = PrepareData(Train_en, Train_cn, 10000)
    print('可以运行')

