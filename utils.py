import os
import pickle
import numpy as np


def softmax(a):
    tmp = np.max(a)
    exp_a = np.exp(a - tmp)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def cos_lr(t, T, lr):
    new_lr = 0.5 * (1 + np.cos(t*np.pi/T)) * lr
    return new_lr


def cross_entropy_loss(x, y, reduction='mean'):
    loss = []
    for i in range(y.shape[0]):
        loss_i = -x[i, y[i]] + np.log(np.sum(np.exp(x[i])))
        loss.append(loss_i)
    dif = np.array(loss).reshape([y.shape[0], -1])
    # print(dif)

    if reduction == 'mean':
        return np.mean(dif)
    elif reduction == 'sum':
        return np.sum(dif)
    return dif


def one_hot(x):
    y = np.zeros((10, x.shape[0]))
    for i in range(x.shape[0]):
        y[x[i], i] = 1
    return y


def save(model, path):
    # if os.path.exists(path):
    #     (name, suffix) = os.path.splitext(path)
    #     filename = name + "(1)" + suffix
    f = open(path, 'wb')
    pickle.dump(model, f)
    f.close()


def load(path):
    f = open(path, 'rb')
    network = pickle.load(f)
    f.close()
    return network


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


if __name__ == '__main__':
    x = np.array([1,2,3])
    print(x)
    print(one_hot(x))
