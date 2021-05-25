# 一些工具函数，比如不等长句子的padding

import numpy as np
import torch


# 对一个batch批次(以单词id表示)的数据进行padding填充对齐长度
def seq_padding(X, padding=0):
    # 计算该批次数据各条数据句子长度
    L = [len(x) for x in X]
    # 获取该批次数据最大句子长度
    ML = max(L)
    # 对X中各条数据x进行遍历，如果长度短于该批次数据最大长度ML，则以padding id填充缺失长度ML-len(x)
    # （注意这里默认padding id是0，相当于是拿<UNK>来做了padding）
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


# 生成Mask矩阵
def subsequent_mask(size):
    """
    deocer层self attention需要使用一个mask矩阵，
    :param size: 句子维度
    :return: 右上角(不含对角线)全为False，左下角全为True的mask矩阵
    """
    # Mask out（遮掩） subsequent（后来的） positions.
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)
    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0


# 读取字典文件
def get_word_dict():
    """
    获取中英，word2idx和idx2word字典
    :return: 各个字典
    """
    import csv
    cn_idx2word = {}
    cn_word2idx = {}
    en_idx2word = {}
    en_word2idx = {}
    # 这里对词典的路径进行了修改：
    with open("../Dict/cn_index_dict.csv", 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)
        for l in data:
            cn_idx2word[int(l[0])] = l[1]
            cn_word2idx[l[1]] = int(l[0])
    with open("../Dict/en_index_dict.csv", 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)
        for l in data:
            en_idx2word[int(l[0])] = l[1]
            en_word2idx[l[1]] = int(l[0])

    return cn_idx2word, cn_word2idx, en_idx2word, en_word2idx

