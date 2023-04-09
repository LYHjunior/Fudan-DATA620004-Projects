import numpy as np
from train import *
from model import *


lr_range = [0.05, 0.15]
hidden_range = [100, 300]
decay_range = [1e-5, 1e-4]
num_sample = 4
batch_size = 128
search_epoch = 10


def search(train_data, train_labels, test_data, test_labels):
    lr = np.random.uniform(lr_range[0], lr_range[1], num_sample)
    hidden = np.random.randint(hidden_range[0], hidden_range[1], num_sample)
    exp_decay = np.random.uniform(np.log(decay_range[0]), np.log(decay_range[1]), num_sample)
    decay = np.exp(exp_decay)
    best_prec1 = 0
    best_hidden, best_lr, best_decay = 0, 0, 0
    data_len = len(train_data)
    iteration = int(data_len/batch_size) + 1
    for i in hidden:
        model = MyNet(i, batch_size)
        for j in lr:
            for k in decay:
                current_lr = j
                for e in range(search_epoch):
                    # train
                    _ = train(model, train_data, train_labels, iteration, batch_size, data_len, current_lr, k)
                    current_lr = cos_lr(e, search_epoch, j)

                # test
                loss, prec1 = test(model, test_data, test_labels)
                if best_prec1 < prec1:
                    best_hidden, best_lr, best_decay = i, j, k

    return best_hidden, best_lr, best_decay


if __name__ == '__main__':
    decay = np.random.uniform(np.log(decay_range[0]), np.log(decay_range[1]), num_sample)
    decay = np.exp(decay)
    print(decay)
