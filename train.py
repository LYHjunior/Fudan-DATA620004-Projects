import numpy as np
from utils import *


def test(model, test_data, test_labels):
    output = model.forward(test_data)
    loss = cross_entropy_loss(output.T, test_labels)
    prec1 = np.sum(np.argmax(output, axis=0) == test_labels) / len(test_labels)
    print('Accuracy {0}, Loss {1}'.format(prec1, loss))
    return loss, prec1


def train(model, train_data, train_labels, iteration, batch_size, data_len, lr, reg):
    train_loss = AverageMeter()
    for iter in range(iteration):
        current_train_data = train_data[:, iter * batch_size:min((iter + 1) * batch_size, data_len)]
        # current_train_data shape: (784, batch size)
        current_train_labels = train_labels[iter * batch_size:min((iter + 1) * batch_size, data_len)]
        output = model.forward(current_train_data)
        loss = cross_entropy_loss(output.T, current_train_labels)
        train_loss.update(loss.item(), len(current_train_labels))

        model.backward(current_train_labels, lr, reg)
    return train_loss.avg
