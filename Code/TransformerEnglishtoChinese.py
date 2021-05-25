# 用Transformer实现机器翻译：

import os
import torch
import warnings
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

from data_pre import PrepareData
from model import make_model
from test import Test
from create_dict import Make_dict
from train import Train
from setting import MAX_LENGTH, DEVICE, LAYERS, D_MODEL, D_FF, \
    DROPOUT, H_NUM, Train_en, Train_cn, Test_en, Test_cn

# 关掉Warning
warnings.filterwarnings('ignore')

# 固定随机数种子：
torch.manual_seed(10)

# （1）如果没有字典，创建一个字典，如果想覆盖原字典，运行create_dict.py
if not os.path.exists('../Dict/cn_index_dict.csv'):
    Make_dict()  # 根据训练集的数据创建字典

# （2）数据读取--训练集：
Test_num = 10000  # 要翻译的句子数量
data_Test = PrepareData(Test_en, Test_cn, Test_num, sort=False)  # 不对句子按长度进行排序

# （3）如果没有模型，训练一个新模型，如果想覆盖原模型，运行train.py
if not os.path.exists('../Model/model.pt'):
    Train(Train_num=200000, Valid_num=100000)

# 模型的初始化
model = make_model(
    data_Test.En_vocab,  # 英文词典数
    data_Test.Cn_vocab,  # 中文词典数
)
# 导入模型参数：
model.load_state_dict(torch.load('../Model/model.pt', map_location=torch.device('cpu')))

# （4）模型测试：
[Refer, Candi] = Test(data_Test, model)
# 返回真实结果 Refer=[[['a', 'b', 'c']], [['1', '2', '3']]]和预测结果 Candi=[['c', 'd'], ['3', '4']]

# （5）计算BLEU值：
BLEU1 = corpus_bleu(Refer, Candi, weights=(1, 0, 0, 0))
BLEU2 = corpus_bleu(Refer, Candi, weights=(0, 1, 0, 0))
BLEU3 = corpus_bleu(Refer, Candi, weights=(0, 0, 1, 0))
BLEU4 = corpus_bleu(Refer, Candi, weights=(0, 0, 0, 1))
print('************* BLEU *************')
print('共评测了{}个结果'.format(Test_num))
print('BLEU1: {:.4f}'.format(BLEU1))
print('BLEU2: {:.4f}'.format(BLEU2))
print('BLEU3: {:.4f}'.format(BLEU3))
print('BLEU4: {:.4f}'.format(BLEU4))

