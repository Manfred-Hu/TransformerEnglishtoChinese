# 模型训练：

import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt

# 从自定义函数中调用：
from data_pre import PrepareData
from setting import EPOCHS
from model import SimpleLossCompute, LabelSmoothing, NoamOpt
from model import make_model
from setting import LAYERS, D_MODEL, D_FF, DROPOUT, H_NUM,\
    Train_en, Train_cn, Valid_en, Valid_cn
# 块的个数，embedding维度，全连接层的维度，dropout率，multi-head的个数，
# 训练集、测试集的中英文保存路径


def run_epoch(data, model, loss_compute, epoch):
    # loss_compute: loss_compute函数

    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.

    for i, batch in enumerate(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens  # 实际的词数

        if i % int(len(data)/5) == 1:
            elapsed = time.time() - start
            print("Epoch %d Batch: %d Loss: %f Tokens per Sec: %0.3f" % (
            epoch + 1, i - 1, loss / batch.ntokens, (tokens.float() / elapsed / 1000.)))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens

# 训练并保存最优模型：
def train(data_Train, data_Valid, model, criterion, optimizer):

    # 记录训练过程中的训练误差，验证误差：
    train_loss_all = []
    valid_loss_all = []

    # 初始化模型在验证集上的最优Loss为一个较大值
    best_valid_loss = 1e5

    for epoch in range(EPOCHS):
        """
        每次迭代一次就在验证集上验证loss
        """
        time_epoch_start = time.time()
        model.train()
        train_loss = run_epoch(data_Train.data, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch)
        train_loss_all.append(train_loss)
        model.eval()

        # 在dev集上进行loss评估
        print('>>>>> Evaluate')
        valid_loss = run_epoch(data_Valid.data, model, SimpleLossCompute(model.generator, criterion, None), epoch)
        valid_loss_all.append(valid_loss)
        print('<<<<< Evaluate loss: %f' % valid_loss)
        # 如果当前epoch的模型在验证集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), '../Model/model.pt')  # 保存模型中的参数
            best_valid_loss = valid_loss
            print('****** Save model done... ******')
        print('第{}轮用时：{:.4f}秒\n'.format(epoch+1, time.time()-time_epoch_start))

    return train_loss_all, valid_loss_all


def Train(Train_num, Valid_num):
    print('**************** 数据准备(训练集+验证集) ****************\n')
    # 数据准备
    data_Train = PrepareData(Train_en, Train_cn, Train_num)  # 只读取文件的前10000行数据
    data_Valid = PrepareData(Valid_en, Valid_cn, Valid_num)

    # 模型的初始化
    model = make_model(
        data_Train.En_vocab,  # 英文词典数--实际的英文词典数与设定值可能不一样
        data_Train.Cn_vocab,  # 中文词典数
    )

    print('**************** 开始训练 ****************')
    train_start = time.time()  # 开始计时
    # 损失函数：
    criterion = LabelSmoothing(data_Train.Cn_vocab, padding_idx=0, smoothing=0.0)
    # 优化器：
    optimizer = NoamOpt(D_MODEL, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    if not os.path.exists('../Model'):
        os.mkdir('../Model')

    [train_loss_all, valid_loss_all] = train(data_Train, data_Valid, model, criterion, optimizer)

    print(f'<<<训练结束, 共花费时间 {time.time() - train_start:.4f}秒')

    # 绘制训练集、验证集的loss随训练过程的变化：
    x_label = np.linspace(1, EPOCHS, EPOCHS)
    plt.figure()
    plt.plot(x_label, train_loss_all, color='b', label='Train loss')
    plt.plot(x_label, valid_loss_all, color='r', label='Valid loss')
    plt.title('Loss via epoch')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()

    if not os.path.exists('../Fig'):
        os.mkdir('../Fig')
    plt.savefig('../Fig/Loss.png')   # 在plt.show()之前保存图片
    plt.show()


if __name__ == '__main__':
    Train(Train_num=20000, Valid_num=10000)

