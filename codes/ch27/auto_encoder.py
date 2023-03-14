#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: auto_encoder.py
@time: 2023/3/14 14:13
@project: statistical-learning-method-solutions-manual
@desc: 习题27.3 2层卷积神经网络编码器和2层卷积神经网络解码器组成的自动编码器
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchviz import make_dot


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # 2层卷积神经网络编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU()
        )
        # 2层卷积神经网络解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, 3, 2, 1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def save_model_structure(model, device):
    x = torch.randn(1, 1, 28, 28).requires_grad_(True).to(device)
    y = model(x)
    vise = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
    vise.format = "png"
    vise.directory = "./data"
    vise.view()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 使用MNIST数据集
    train_set = mnist.MNIST('./data', transform=transforms.ToTensor(), train=True, download=True)
    test_set = mnist.MNIST('./data', transform=transforms.ToTensor(), train=False, download=True)
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=8, shuffle=False)

    model = AutoEncoder().to(device)

    # 设置损失函数
    criterion = nn.MSELoss()
    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    # 模型训练
    EPOCHES = 10
    for epoch in range(EPOCHES):
        for img, _ in tqdm.tqdm(train_dataloader):
            optimizer.zero_grad()

            img = img.to(device)
            out = model(img)
            loss = criterion(out, img)
            loss.backward()

            optimizer.step()

    # 将生成图片和原始图片进行对比
    for i, data in enumerate(test_dataloader):
        img, _ = data
        img = img.to(device)
        model = model.to(device)
        img_new = model(img).detach().cpu().numpy()
        img = img.cpu().numpy()
        plt.figure(figsize=(8, 2))
        for j in range(8):
            plt.subplot(2, 8, j + 1)
            plt.axis('off')
            plt.imshow(img_new[j].squeeze())
            plt.subplot(2, 8, 8 + j + 1)
            plt.axis('off')
            plt.imshow(img[j].squeeze())
        if i >= 2:
            break

    save_model_structure(model, device)
