# Hyper parameters

import torch

UNK = 0
PAD = 1
BATCH_SIZE = 64
# 数据文件：  -- 这里对数据的路径进行了修改
Train_en = '../中英文翻译数据集/train_en'  # 训练集
Train_cn = '../中英文翻译数据集/train_cn'
Valid_en = '../中英文翻译数据集/valid_en'  # 验证集
Valid_cn = '../中英文翻译数据集/valid_cn'
Test_en = '../中英文翻译数据集/test_en'  # 测试集
Test_cn = '../中英文翻译数据集/test_cn'

LAYERS = 6  # encoder和decoder层数
D_MODEL = 512  # embedding 维度
D_FF = 1024  # feed forward第一个全连接层维数
H_NUM = 8  # multi head attention hidden个数
DROPOUT = 0.1  # dropout比例
EPOCHS = 20
MAX_LENGTH = 60   # 翻译结果的长度
EN_VOCAB = 20000  # 英文的单词数
CN_VOCAB = 10000  # 中文的单词数

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
