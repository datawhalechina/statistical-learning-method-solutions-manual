#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: bi-lstm-text-classification.py
@time: 2023/3/15 14:30
@project: statistical-learning-method-solutions-manual
@desc: 习题27.1 基于双向LSTM的预训练语言模型，假设下游任务是文本分类
"""
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载AG_NEWS数据集
train_iter, test_iter = AG_NEWS(root='./data')
# 定义tokenizer
tokenizer = get_tokenizer('basic_english')


# 定义数据处理函数
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


# 构建词汇表
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# 将数据集映射到MapStyleDataset格式
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
# 划分验证集
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

# 设置文本和标签的处理函数
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1


def collate_batch(batch):
    """
    对数据集进行数据处理
    """
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


# 构建数据集的数据加载器
BATCH_SIZE = 256
train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)


class TextClassifier(nn.Module):
    """
    基于双向LSTM的文本分类模型
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, sparse=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        x, _ = self.lstm(embedded)
        x = self.fc(x)
        return x


# 设置超参数
EMBED_DIM = 64
HIDDEN_DIM = 64
NUM_CLASSES = 4
LEARNING_RATE = 1e-2
NUM_EPOCHS = 10

# 创建模型、优化器和损失函数
model = TextClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train(dataloader):
    """
    模型训练
    """
    model.train()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()


def evaluate(dataloader):
    """
    模型验证
    """
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.1f}% '.format(epoch,
                                            time.time() - epoch_start_time,
                                            accu_val * 100))
    print('-' * 59)

# 新闻的分类标签
ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}


def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1


# 预测一个文本的类别
ex_text_str = """
Our younger Fox Cubs (Y2-Y4) also had a great second experience of swimming competition in February when they travelled 
over to NIS at the end of February to compete in the SSL Development Series R2 event. For students aged 9 and under 
these SSL Development Series events are a great introduction to competitive swimming, focussed on fun and participation 
whilst also building basic skills and confidence as students build up to joining the full SSL team in Year 5 and beyond.
"""
model = model.to("cpu")
print("This is a %s news" % ag_news_label[predict(ex_text_str, text_pipeline)])
