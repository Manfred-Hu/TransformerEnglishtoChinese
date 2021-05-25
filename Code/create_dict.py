# 生成字典文件：

import os
import csv
import torch
import numpy as np
from nltk import word_tokenize
from collections import Counter

from setting import UNK, PAD, DEVICE, EN_VOCAB, CN_VOCAB, Train_en, Train_cn


# 数据准备和生成字典整合在一起了：
class Create_dict:
    # 输入训练集、测试集的中英文文件路径：
    def __init__(self, train_file_en, train_file_cn):
        # 读取数据 并分词
        self.train_en, self.train_cn = self.load_data(train_file_en, train_file_cn)

        # 构建单词表
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(self.train_en, EN_VOCAB)  # 输入要保存的字典的最大长度
        print('英文字典的长度为：{}'.format(self.en_total_words))
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(self.train_cn, CN_VOCAB)
        print('中文字典的长度为：{}'.format(self.cn_total_words))

    # 数据加载：中文与英文是不一样的。
    def load_data(self, path_en, path_cn):
        """
        读取翻译前(英文)和翻译后(中文)的数据文件
        每条数据都进行分词，然后构建成包含起始符(BOS)和终止符(EOS)的单词(中文为字符)列表
        形式如：en = [['BOS', 'i', 'love', 'you', 'EOS'], ['BOS', 'me', 'too', 'EOS'], ...]
                cn = [['BOS', '我', '爱', '你', 'EOS'], ['BOS', '我', '也', '是', 'EOS'], ...]
        """
        en = []
        cn = []
        # （1）英文：
        with open(path_en, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()  # 去除字符串两边的空格（包含\n,\r,\t）
                en.append(["BOS"] + word_tokenize(line) + ["EOS"])  # 区分大小写

        # （2）中文：
        with open(path_cn, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                cn.append(["BOS"] + word_tokenize(" ".join([w for w in line])) + ["EOS"])
                                                  # 在中文字之间加空格，能够被word_tokenize识别

        return en, cn

    # 构建字典，一个正向的，一个反向的
    def build_dict(self, sentences, max_words):
        """
        传入load_data构造的分词后的列表数据
        构建词典(key为单词，value为id值)
        """
        # 对数据中所有单词进行计数
        word_count = Counter()

        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1
        # 只保留最高频的前max_words数的单词构建词典
        # 并添加上UNK和PAD两个单词，对应id已经初始化设置过
        """
        most_common(max_words)
        返回word_count中的前max_words个单词
        返回类型：[('a':3),('b':2),...]
        """
        ls = word_count.most_common(max_words)
        # 统计词典的总词数
        # 加2是添加U-unknow和padding符号
        total_words = len(ls) + 2

        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}   # 所有词的index都加2
        word_dict['UNK'] = UNK   # UNK=0
        word_dict['PAD'] = PAD   # PAD=1
        # 再构建一个反向的词典，供id转单词使用
        index_dict = {v: k for k, v in word_dict.items()}

        return word_dict, total_words, index_dict

    # 将字典保存为csv文件：
    def save_to_file(self):
        """
        保存en_word_dict，cn_word_dict,
            en_index_dict, cn_index_dict
            为csv文件,对应的单词和id的映射关系，只需要在运行api文件之前运行就ok。
        """
        data_list = [self.cn_word_dict, self.en_word_dict, self.cn_index_dict, self.en_index_dict]
        file_name_list = ["cn_word_dict", "en_word_dict", "cn_index_dict", "en_index_dict"]
        # 如果保存字典的文件夹不存在，就创建一个：
        if not os.path.exists('../Dict'):
            os.mkdir('../Dict')

        for i, data in enumerate(data_list):
            with open('../Dict/'+file_name_list[i]+'.csv', 'a+', encoding='utf-8',
                      newline='') as f:
                writer = csv.writer(f)
                for k, v in data.items():
                    writer.writerow([k, v])
        print('文件保存成功')


def Make_dict():
    print('**************** 创建字典 ****************')
    Dict = Create_dict(Train_en, Train_cn)
    Dict.save_to_file()


if __name__ == '__main__':
    Make_dict()  # 创建并保存字典，会覆盖已有的字典
