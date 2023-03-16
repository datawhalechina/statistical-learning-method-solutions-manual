#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: cnn-text-classification.py
@time: 2023/3/16 16:22
@project: statistical-learning-method-solutions-manual
@desc: 习题24.7 基于CNN的自然语言句子分类模型
"""
import time
import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from torchtext.data import get_tokenizer, to_map_style_dataset
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
train_dataset = list(to_map_style_dataset(train_iter))
test_dataset = list(to_map_style_dataset(test_iter))
# 划分验证集
num_train = int(len(train_dataset) * 0.9)
train_dataset, val_dataset = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_batch(batch):
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


BATCH_SIZE = 64

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)


class CNN_Text(nn.Module):
    def __init__(self, vocab_size, embed_dim, class_num=4, dropout=0.5, kernel_size: list = None):
        super(CNN_Text, self).__init__()
        if kernel_size is None:
            kernel_size = [3, 4, 5]
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=1, out_channels=256, kernel_size=k) for k in kernel_size])
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * len(kernel_size), 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, class_num)
        )

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        embedded = embedded.unsqueeze(0)
        embedded = embedded.permute(1, 0, 2)
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(nn.functional.relu(conv(embedded)))
        pooled_outputs = []
        for conv_output in conv_outputs:
            pooled = nn.functional.max_pool1d(conv_output, conv_output.shape[-1]).squeeze(-1)
            pooled_outputs.append(pooled)
        cat = torch.cat(pooled_outputs, dim=-1)
        return self.fc(cat)


# 设置超参数
vocab_size = len(vocab)
embed_dim = 64
class_num = len(set([label for label, _ in train_iter]))
lr = 1e-3
dropout = 0.5
epochs = 10

# 创建模型、优化器和损失函数
model = CNN_Text(vocab_size, embed_dim, class_num, dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


def train(dataloader):
    model.train()

    for label, text, offsets in dataloader:
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()


def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for label, text, offsets in dataloader:
            predicted_label = model(text, offsets)
            criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


max_accu = 0
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.1f}% '.format(epoch,
                                            time.time() - epoch_start_time,
                                            accu_val * 100))
    print('-' * 59)
    if max_accu < accu_val:
        best_model = model
        max_accu = accu_val

# 在测试集上测试模型
test_acc = 0.0
with torch.no_grad():
    for label, text, offsets in test_dataloader:
        output = best_model(text, offsets)
        pred = output.argmax(dim=1)
        test_acc += (pred == label).sum().item()
test_acc /= len(test_dataset)

print(f"Test Acc: {test_acc * 100 :.1f}%")

ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}


def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = best_model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1


ex_text_str = """
Our younger Fox Cubs (Y2-Y4) also had a great second experience of swimming competition in February when they travelled 
over to NIS at the end of February to compete in the SSL Development Series R2 event. For students aged 9 and under 
these SSL Development Series events are a great introduction to competitive swimming, focussed on fun and participation 
whilst also building basic skills and confidence as students build up to joining the full SSL team in Year 5 and beyond.
"""
model = best_model.to("cpu")
print("This is a %s news" % ag_news_label[predict(ex_text_str, text_pipeline)])
