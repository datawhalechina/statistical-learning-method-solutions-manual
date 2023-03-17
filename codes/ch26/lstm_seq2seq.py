#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lstm_seq2seq.py
@time: 2023/3/17 18:37
@project: statistical-learning-method-solutions-manual
@desc: 习题26.1 4层LSTM组成的序列到序列的基本模型
"""

import torch
from torch import nn
import numpy as np


class S2SEncoder(nn.Module):
    r"""由LSTM组成的序列到序列编码器。

    Args:
        inp_size: 嵌入层的输入维度
        embed_size: 嵌入层的输出维度
        num_hids: LSTM隐层向量维度
        num_layers: LSTM层数，本题目设置为4
    """

    def __init__(self, inp_size, embed_size, num_hids,
                 num_layers, dropout=0, **kwargs):
        super(S2SEncoder, self).__init__(**kwargs)

        self.embed = nn.Embedding(inp_size, embed_size)
        self.rnn = nn.LSTM(embed_size, num_hids, num_layers,
                           dropout=dropout)

    def forward(self, inputs):
        # inputs.shape(): (seq_length, embed_size)
        inputs = self.embed(inputs)

        # output.shape(): (seq_length, num_hids)
        # states.shape(): (num_layers, num_hids)
        output, state = self.rnn(inputs)

        return output, state


class S2SDecoder(nn.Module):
    r"""由LSTM组成的序列到序列解码器。

    Args:
        inp_size: 嵌入层的输入维度。
        embed_size: 嵌入层的输出维度。
        num_hids: LSTM 隐层向量维度。
        num_layers: LSTM 层数，本题目设置为4。
    """

    def __init__(self, inp_size, embed_size, num_hids,
                 num_layers, dropout=0, **kwargs):
        super(S2SDecoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.embed = nn.Embedding(inp_size, embed_size)
        # 解码器 LSTM 的输入，由目标序列的嵌入向量和编码器的隐层向量拼接而成。
        self.rnn = nn.LSTM(embed_size + num_hids, num_hids, num_layers,
                           dropout=dropout)

        self.linear = nn.Linear(num_hids, inp_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1][-1]

    def forward(self, inputs, state):
        # inputs.shape(): (seq_length, embed_size)
        inputs = self.embed(inputs)

        # 广播 context，使其具有与 inputs 相同的长度
        # context.shape(): (seq_length, num_layers, embed_size)
        context = state[-1].repeat(inputs.shape[0], 1, 1)
        inputs = torch.cat((inputs, context), 2)
        # output.shape(): (seq_length, num_hids)
        output, _ = self.rnn(inputs)

        output = self.linear(output)

        return output


class EncoderDecoder(nn.Module):
    r"""基于 LSTM 的序列到序列模型。

    Args:
        encoder: 编码器。
        decoder: 解码器。
    """

    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_inp, dec_inp):
        enc_out = self.encoder(enc_inp)
        dec_state = self.decoder.init_state(enc_out)

        return self.decoder(dec_inp, dec_state)


if __name__ == '__main__':
    # 搭建一个4层LSTM构成的序列到序列模型，进行前向计算
    inp_size, embed_size, num_hids, num_layers = 10, 8, 16, 4
    encoder = S2SEncoder(inp_size, embed_size, num_hids, num_layers)
    decoder = S2SDecoder(inp_size, embed_size, num_hids, num_layers)
    model = EncoderDecoder(encoder, decoder)

    enc_inp_seq = "I love you !"
    dec_inp_seq = "我 爱 你 ！"
    enc_inp, dec_inp = [], []

    # 自己构造的的词典
    word2vec = {"I": [1, 0, 0, 0],
                "love": [0, 1, 0, 0],
                "you": [0, 0, 1, 0],
                "!": [0, 0, 0, 1],
                "我": [1, 0, 0, 0],
                "爱": [0, 1, 0, 0],
                "你": [0, 0, 1, 0],
                "！": [0, 0, 0, 1]}

    for word in enc_inp_seq.split():
        enc_inp.append(word2vec[word])

    enc_inp = torch.tensor(enc_inp)

    for word in dec_inp_seq.split():
        dec_inp.append(word2vec[word])

    dec_inp = torch.tensor(dec_inp)
    output = model(enc_inp, dec_inp)
