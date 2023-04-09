import numpy as np
from utils import one_hot


def relu(x):
    x = np.maximum(0, x)
    return x


def drelu(x):
    dx = np.zeros_like(x)
    dx[x > 0] = 1
    return dx


class MyNet:
    def __init__(self, size, batch_size):
        self.w1 = np.random.randn(size, 784) / np.sqrt(784/2)
        self.w2 = np.random.randn(10, size) / np.sqrt(size/2)
        self.b1 = np.random.randn(size, 1) / np.sqrt(size/2)
        self.b2 = np.random.randn(10, 1) / np.sqrt(10/2)
        self.batch_size = batch_size

    def forward(self, input):
        # input shape: (784, batch size)
        # act shape: (args.size, batch size)
        # output shape: (10, batch size)
        pre_act = np.matmul(self.w1, input) + self.b1
        act = relu(pre_act)
        output = np.matmul(self.w2, act) + self.b2

        self.input_data = input
        self.act = act
        self.output = output

        return output

    def backward(self, label, lr, reg):
        # label shape: (batch size, )
        # one_hot(label) shape: (10, batch size)
        # self.output shape: (10, batch size)
        # dout shape: (10, batch size)
        dout = np.exp(self.output) / np.sum(np.exp(self.output), axis=0).reshape(1, -1) - one_hot(label)
        dw1 = np.matmul((np.matmul(self.w2.T, dout) * drelu(self.act)), self.input_data.T) / self.batch_size
        dw2 = np.matmul(dout, self.act.T) / self.batch_size
        db1 = np.mean(np.matmul(self.w2.T, dout) * drelu(self.act), axis=1).reshape(-1, 1)
        db2 = np.mean(dout, axis=1).reshape(-1, 1)

        self.w1 -= lr * dw1 + reg * dw1
        self.w2 -= lr * dw2 + reg * dw2
        self.b1 -= lr * db1 + reg * db1
        self.b2 -= lr * db2 + reg * db2