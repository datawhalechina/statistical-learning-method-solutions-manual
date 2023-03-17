#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: feedforward_nn_backpropagation.py
@time: 2023/3/17 18:32
@project: statistical-learning-method-solutions-manual
@desc: 习题23.3 自编程实现前馈神经网络的反向传播算法，使用MNIST数据构建手写数字识别网络
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from tqdm import tqdm

np.random.seed(2023)


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # 网络层的神经元个数，其中第一层和第二层是隐层
        self.layers = layers
        # 学习率
        self.alpha = alpha
        # 权重
        self.weights = []
        # 偏置
        self.biases = []
        # 初始化权重和偏置
        for i in range(1, len(layers)):
            self.weights.append(np.random.randn(layers[i - 1], layers[i]))
            self.biases.append(np.random.randn(layers[i]))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, inputs):
        """
        （1）正向传播
        """
        self.activations = [inputs]
        self.weighted_inputs = []
        for i in range(len(self.weights)):
            weighted_input = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.weighted_inputs.append(weighted_input)
            # 得到各层的输出h
            activation = self.sigmoid(weighted_input)
            self.activations.append(activation)

        return self.activations[-1]

    def backpropagate(self, expected):
        """
        （2）反向传播
        """
        # 计算各层的误差
        errors = [expected - self.activations[-1]]
        # 计算各层的梯度
        deltas = [errors[-1] * self.sigmoid_derivative(self.activations[-1])]

        for i in range(len(self.weights) - 1, 0, -1):
            error = deltas[-1].dot(self.weights[i].T)
            errors.append(error)
            delta = errors[-1] * self.sigmoid_derivative(self.activations[i])
            deltas.append(delta)
        deltas.reverse()

        for i in range(len(self.weights)):
            # 更新参数
            self.weights[i] += self.alpha * np.array([self.activations[i]]).T.dot(np.array([deltas[i]]))
            self.biases[i] += self.alpha * np.sum(deltas[i], axis=0)

    def train(self, inputs, expected_outputs, epochs):
        for i in tqdm(range(epochs)):
            for j in range(len(inputs)):
                self.feedforward(inputs[j])
                self.backpropagate(expected_outputs[j])


if __name__ == '__main__':
    # 加载MNIST手写数字数据集
    mnist = fetch_openml('mnist_784', parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')

    # 划分训练集和测试集
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    X = np.array(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练神经网络，其中第一层和第二层各有100个神经元和50个神经元
    nn = NeuralNetwork([784, 100, 50, 10], alpha=0.1)
    nn.train(X_train, y_train, epochs=10)

    # 使用测试集对模型进行评估
    correct = 0

    for i in range(len(X_test)):
        output = nn.feedforward(X_test[i])
        prediction = np.argmax(output)
        actual = np.argmax(y_test[i])
        if prediction == actual:
            correct += 1

    accuracy = correct / len(X_test) * 100
    print("Accuracy: {:.2f} %".format(accuracy))
