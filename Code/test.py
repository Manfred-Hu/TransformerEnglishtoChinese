
import os
import torch
import time
from torch.autograd import Variable
import numpy as np

from utils import subsequent_mask
from setting import MAX_LENGTH, DEVICE, LAYERS, D_MODEL, D_FF, \
    DROPOUT, H_NUM, Test_en, Test_cn
from model import make_model
from data_pre import PrepareData


# 用greedy方法进行预测：
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    # 先用encoder进行encode
    memory = model.encode(src, src_mask)
    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # 遍历输出的长度下标
    for i in range(max_len - 1):
        # decode得到隐层表示
        out = model.decode(memory,
                           src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, -1])
        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


# 用训练好的模型对数据进行预测（翻译）：
def Test(data, model):
    # 正确结果，翻译结果：
    Refer = []  # Refer = [[['a', 'b', 'c']], [['1', '2', '3']]]
    Candi = []  # Candi = [['c', 'd'], ['3', '4']]

    Out_num = min(50, len(data.data_en))  # 写入文件中的翻译例子的数量

    # 开始计时：
    time_start = time.time()

    print('************* 开始翻译 *************')

    # 将翻译结果的前50条存到'../Trans_result/Trans_result.txt'中
    if not os.path.exists('../Trans_result'):
        os.mkdir('../Trans_result')

    # 将翻译结果写入文件中：
    f = open('../Trans_result/Trans_result.txt', 'w', encoding='utf-8')
    # 梯度清零
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for i in range(len(data.data_en)):
            if Out_num > 0:
                # 将英文原句写入文件：
                en_sent = " ".join([data.en_index_dict[w] for w in data.data_en[i][1:len(data.data_en[i])-1]])  # 不要开头的BOS和结尾的EOS
                f.write('第{}个句子：\n'.format(i+1))
                f.write("Source Language: " + en_sent + '\n')
                # 将参考翻译写入文件
                cn_sent = "".join([data.cn_index_dict[w] for w in data.data_cn[i][1:len(data.data_cn[i])-1]])
                f.write("Target Language: " + cn_sent + '\n')

            # 添加参考翻译：
            Refer.append(
                [[data.cn_index_dict[w] for w in data.data_cn[i][1:len(data.data_cn[i]) - 1]]])

            # 将当前以单词id表示的英文句子数据转为tensor，并放如DEVICE中
            src = torch.from_numpy(np.array(data.data_en[i])).long().to(DEVICE)
            # 增加一维
            src = src.unsqueeze(0)
            # 设置attention mask
            src_mask = (src != 0).unsqueeze(-2)
            # 用训练好的模型进行decode预测
            out = greedy_decode(model, src, src_mask, max_len=MAX_LENGTH, start_symbol=data.cn_word_dict["BOS"])
            # 初始化一个用于存放模型翻译结果句子单词的列表
            translation = []
            # 遍历翻译输出字符的下标（注意：开始符"BOS"的索引0不遍历）
            for j in range(1, out.size(1)):  # out的size是确定的，[1, MAX_LENGTH]
                # 获取当前下标的输出字符
                sym = data.cn_index_dict[out[0, j].item()]
                # 如果输出字符不为'EOS'终止符，则添加到当前句子的翻译结果列表
                if sym != 'EOS':
                    translation.append(sym)
                # 否则终止遍历
                else:
                    break
            if Out_num > 0:
                # 将翻译结果写入文件：
                tran_result = ''.join(translation)
                f.write('Translation Result: ' + tran_result + '\n')
                Out_num -= 1

            # 添加翻译结果：
            Candi.append(translation)

            if i % int(len(data.data_en) / 5) == 1:
                print('已翻译{}条句子，用时{:.3f}s'.format(i, time.time()-time_start))
    f.close()
    print('************* 翻译结果输出成功 ************\n')
    return Refer, Candi


if __name__ == '__main__':
    # 测试集数据读取：
    Test_num = 100  # 要翻译的句子数量
    data_Test = PrepareData(Test_en, Test_cn, Test_num, sort=False)  # 不对句子按长度进行排序

    # 模型的初始化
    model = make_model(
        data_Test.En_vocab,  # 英文词典数--实际的英文词典数与设定值可能不一样
        data_Test.Cn_vocab,  # 中文词典数
    )
    # 导入模型参数：
    model.load_state_dict(torch.load('../Model/model.pt', map_location=torch.device('cpu')))
    # 模型测试：
    [Refer, Candi] = Test(data_Test, model)

    # 计算BLEU值：
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

