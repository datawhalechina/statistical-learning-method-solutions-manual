#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: cnn_seq2seq.py
@time: 2023/3/17 18:44
@project: statistical-learning-method-solutions-manual
@desc: 习题26.4 基于CNN的序列到序列模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    r"""序列到序列 CNN 编码器。

    Args:
        inp_dim: 嵌入层的输入维度。
        emb_dim: 嵌入层的输出维度。
        hid_dim: CNN 隐层向量维度。
        num_layers: CNN 层数。
        kerner_size: 卷积核大小。
    """

    def __init__(self, inp_dim, emb_dim, hid_dim,
                 num_layers, kernel_size):
        super().__init__()

        self.embed = nn.Embedding(inp_dim, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, int(hid_dim / 2))
        self.hid2emb = nn.Linear(int(hid_dim / 2), emb_dim)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=emb_dim,
                                              out_channels=hid_dim,
                                              kernel_size=kernel_size,
                                              padding=(kernel_size - 1) // 2)
                                    for _ in range(num_layers)])

    def forward(self, inputs):
        # inputs.shape(): (src_len, inp_dim)
        # conv_inp.shape(): (src_len, emb_dim)
        conv_inp = self.embed(inputs).permute(0, 2, 1)

        for _, conv in enumerate(self.convs):
            # 进行卷积运算
            # conv_out.shape(): (src_len, hid_dim)
            conv_out = conv(conv_inp)

            # 经过激活函数
            conved = F.glu(conv_out, dim=1)

            # 残差连接运算
            conved = self.hid2emb(conved.permute(0, 2, 1)).permute(0, 2, 1)
            conved = conved + conv_inp
            conv_inp = conved

        # 卷积输出与词嵌入 element-wise 点加进行注意力运算
        # combined.shape(): (src_len, emb_dim)
        combined = conved + conv_inp

        return conved, combined


class CNNDecoder(nn.Module):
    r"""序列到序列 CNN 解码器。

    Args:
        out_dim: 嵌入层的输入维度。
        emb_dim: 嵌入层的输出维度。
        hid_dim: CNN 隐层向量维度。
        num_layers: CNN 层数。
        kernel_size: 卷积核大小。
    """

    def __init__(self, out_dim, emb_dim, hid_dim,
                 num_layers, kernel_size, trg_pad_idx):
        super().__init__()

        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx

        self.embed = nn.Embedding(out_dim, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, int(hid_dim / 2))
        self.hid2emb = nn.Linear(int(hid_dim / 2), emb_dim)

        self.attn_hid2emb = nn.Linear(int(hid_dim / 2), emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, int(hid_dim / 2))

        self.fc_out = nn.Linear(emb_dim, out_dim)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=emb_dim,
                                              out_channels=hid_dim,
                                              kernel_size=kernel_size)
                                    for _ in range(num_layers)])

    def calculate_attention(self, embed, conved, encoder_conved, encoder_combined):
        # embed.shape(): (trg_len, emb_dim)
        # conved.shape(): (hid_dim, trg_len)
        # encoder_conved.shape(), encoder_combined.shape(): (src_len, emb_dim)
        # 进行注意力层第一次线性运算调整维度
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1)).permute(0, 2, 1)

        # conved_emb.shape(): (trg_len, emb_dim])
        combined = conved_emb + embed
        # print(combined.size(), encoder_conved.size())
        energy = torch.matmul(combined.permute(0, 2, 1), encoder_conved)

        # attention.shape(): (trg_len, emb_dim])
        attention = F.softmax(energy, dim=2)
        attended_encoding = torch.matmul(attention, encoder_combined.permute(0, 2, 1))

        # attended_encoding.shape(): (trg_len, emd_dim)
        # 进行注意力层第二次线性运算调整维度
        attended_encoding = self.attn_emb2hid(attended_encoding)

        # attended_encoding.shape(): (trg_len, hid_dim)
        # 残差计算
        attended_combined = conved + attended_encoding.permute(0, 2, 1)

        return attention, attended_combined

    def forward(self, targets, encoder_conved, encoder_combined):
        # targets.shape(): (trg_len, out_dim)
        # encoder_conved.shape(): (src_len, emb_dim)
        # encoder_combined.shape(): (src_len, emb_dim)
        conv_inp = self.embed(targets).permute(0, 2, 1)

        src_len = conv_inp.shape[0]
        hid_dim = conv_inp.shape[1]

        for _, conv in enumerate(self.convs):
            # need to pad so decoder can't "cheat"
            padding = torch.zeros(src_len, hid_dim,
                                  self.kernel_size - 1).fill_(self.trg_pad_idx)

            padded_conv_input = torch.cat((padding, conv_inp), dim=-1)

            # padded_conv_input = [batch size, hid dim, trg len + kernel size - 1]

            # 经过卷积运算
            conved = conv(padded_conv_input)

            # 经过激活函数
            conved = F.glu(conved, dim=1)

            # 注意力分数计算
            attention, conved = self.calculate_attention(conv_inp, conved,
                                                         encoder_conved,
                                                         encoder_combined)

            # 残差连接计算
            conved = self.hid2emb(conved.permute(0, 2, 1)).permute(0, 2, 1)

            conved = conved + conv_inp
            conv_inp = conved

        output = self.fc_out(conved.permute(0, 2, 1))
        return output, attention


class EncoderDecoder(nn.Module):
    r"""序列到序列 CNN 模型。
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_inp, dec_inp):
        # 编码器，将源句子编码为向量输入解码器进行解码。
        encoder_conved, encoder_combined = self.encoder(enc_inp)

        # 解码器，根据编码器隐藏状态和解码器输入预测下一个单词的概率
        # 注意力层，源句子和目标句子之间进行注意力运算从而对齐
        output, attention = self.decoder(dec_inp, encoder_conved, encoder_combined)

        return output, attention


if __name__ == '__main__':
    # 构建一个基于CNN的序列到序列模型
    inp_dim, out_dim, emb_dim, hid_dim, num_layers, kernel_size = 8, 10, 12, 16, 1, 3
    encoder = CNNEncoder(inp_dim, emb_dim, hid_dim, num_layers, kernel_size)
    decoder = CNNDecoder(out_dim, emb_dim, hid_dim, num_layers, kernel_size, trg_pad_idx=0)
    model = EncoderDecoder(encoder, decoder)

    enc_inp_seq = "I love you"
    dec_inp_seq = "我 爱 你"
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
        pass

    enc_inp = torch.tensor(enc_inp)

    for word in dec_inp_seq.split():
        dec_inp.append(word2vec[word])
        pass
    dec_inp = torch.tensor(dec_inp)

    output = model(enc_inp, dec_inp)
