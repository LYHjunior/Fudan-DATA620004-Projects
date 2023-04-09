import numpy as np
import argparse
import mnist
import matplotlib.pyplot as plt
from utils import *
from model import *
from search import *
from train import train, test

parser = argparse.ArgumentParser(description='Train classification network')
parser.add_argument('--lr', help='learning rate', type=float, default=0.1)
parser.add_argument('--size', help='number of hidden layer neurons', type=int, default=100)
parser.add_argument('--decay', help='weight decay', type=float, default=1e-4)
parser.add_argument('--path', help='data path', type=str, default='./MNIST')
parser.add_argument('--epoch', help='epoch', type=int, default=50)
parser.add_argument('--batch_size', help='batch size', type=int, default=128)
parser.add_argument('--resume', help='log path', type=str, default='')

args = parser.parse_args()


def plot(test_acc, test_loss, train_loss, model):
    plt.plot(test_acc)
    plt.xlabel('Epoch')
    plt.xlabel('Accuracy')
    plt.title('Test Accuracy')
    plt.show()
    plt.savefig('Test_Accuracy')

    plt.plot(test_loss)
    plt.xlabel('Epoch')
    plt.xlabel('Test Loss')
    plt.title('Loss')
    plt.savefig('Test_Loss')
    plt.show()

    plt.plot(train_loss)
    plt.xlabel('Iteration')
    plt.xlabel('Loss')
    plt.title('Training Loss')
    plt.savefig('Training_Loss')
    plt.show()

    plt.imshow(model.w1, cmap='RdBu', interpolation='nearest')
    plt.title('w1')
    plt.savefig('w1')
    plt.show()
    plt.imshow(model.w2, cmap='RdBu', interpolation='nearest')
    plt.title('w2')
    plt.savefig('w2')
    plt.show()


def main():
    train_data, train_labels, test_data, test_labels = mnist.load_mnist()
    train_data, test_data = train_data / 255, test_data / 255
    # train_data shape: (784, 60000)
    # train_labels shape: (60000, )

    if args.resume:
        print('resume from pretrained models...')
        state_dict = load(args.resume)
        hidden_size = state_dict['hidden_size']
        model = MyNet(hidden_size, args.batch_size)

        for name in state_dict['params'].keys():
            setattr(model, name, state_dict['params'][name])
        loss, prec1 = test(model, test_data, test_labels)
    else:
        lr = args.lr
        batch_size = args.batch_size
        decay = args.decay
        data_len = len(train_labels)
        iteration = int(data_len / args.batch_size) + 1
        test_loss = []
        test_acc = []
        train_losses = []
        # args.size, lr, decay = search(train_data, train_labels, test_data, test_labels)
        model = MyNet(args.size, args.batch_size)

        for i in range(args.epoch):
            # train
            train_loss = train(model, train_data, train_labels, iteration, batch_size, data_len, lr, decay)
            train_losses.append(train_loss)
            lr = cos_lr(i, args.epoch, args.lr)

            # test
            loss, prec1 = test(model, test_data, test_labels)
            test_loss.append(loss)
            test_acc.append(prec1)

        dict = {'params': {'w1': model.w1, 'w2': model.w2, 'b1': model.b1, 'b2': model.b2},
                'lr': lr, 'hidden_size': args.size, 'decay': decay}
        save(dict, './model.pkl')
        plot(test_acc, test_loss, train_losses, model)


if __name__ == '__main__':
    main()
