#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: bi-lstm-text-classification.py
@time: 2023/3/15 14:30
@project: statistical-learning-method-solutions-manual
@desc: 习题27.1 基于双向LSTM的ELMo预训练语言模型，假设下游任务是文本分类
"""
import os
import time

import torch
import torch.nn as nn
import wget
from allennlp.modules.elmo import Elmo
from allennlp.modules.elmo import batch_to_ids
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torchtext.datasets import AG_NEWS


def get_elmo_model():
    elmo_options_file = './data/elmo_2x1024_128_2048cnn_1xhighway_options.json'
    elmo_weight_file = './data/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
    url = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
    if (not os.path.exists(elmo_options_file)):
        wget.download(url, elmo_options_file)
    url = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
    if (not os.path.exists(elmo_weight_file)):
        wget.download(url, elmo_weight_file)

    elmo = Elmo(elmo_options_file, elmo_weight_file, 1)
    return elmo


# 加载ELMo模型
elmo = get_elmo_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_pipeline = lambda x: int(x) - 1


def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        text_list.append(_text.split())
    label_list = torch.tensor(label_list, dtype=torch.int64)
    return label_list.to(device), text_list


# 加载AG_NEWS数据集
train_iter, test_iter = AG_NEWS(root='./data')
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = \
    random_split(train_dataset, [num_train, len(train_dataset) - num_train])

BATCH_SIZE = 128
train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_batch)


class TextClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        # 使用预训练的ELMO
        self.elmo = elmo

        # 使用双向LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)

        # 使用线性函数进行文本分类任务
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

        self.dropout = nn.Dropout(0.5)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.uniform_(-initrange, initrange)

    def forward(self, sentence_lists):
        character_ids = batch_to_ids(sentence_lists)
        character_ids = character_ids.to(device)

        embeddings = self.elmo(character_ids)
        embedded = embeddings['elmo_representations'][0]

        x, _ = self.lstm(embedded)
        x = x.mean(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


EMBED_DIM = 256
HIDDEN_DIM = 64
NUM_CLASSES = 4
LEARNING_RATE = 1e-2
NUM_EPOCHS = 1

model = TextClassifier(EMBED_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)


def train(dataloader):
    model.train()

    for idx, (label, text) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            predicted_label = model(text)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

    return total_acc / total_count


# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss().to(device)
# 设置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

ag_news_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}


def predict(text):
    with torch.no_grad():
        output = model([text])
        return output.argmax(1).item() + 1


ex_text_str = """
Our younger Fox Cubs (Y2-Y4) also had a great second experience 
of swimming competition in February when they travelled over to 
NIS at the end of February to compete in the SSL Development 
Series R2 event. For students aged 9 and under these SSL 
Development Series events are a great introduction to 
competitive swimming, focussed on fun and participation whilst 
also building basic skills and confidence as students build up 
to joining the full SSL team in Year 5 and beyond.
"""

print("This is a %s news" % ag_news_label[predict(ex_text_str)])
