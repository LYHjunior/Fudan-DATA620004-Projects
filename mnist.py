import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(path="./MNIST/raw"):
    """Load MNIST data from `path`"""
    kind = 'train'
    train_labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    train_images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    kind = 't10k'
    test_labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    test_images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(train_labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        train_labels = np.fromfile(lbpath, dtype=np.uint8)
    # 读入magic是一个文件协议的描述,也是调用fromfile 方法将字节读入NumPy的array之前在文件缓冲中的item数(n).

    with open(train_images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        train_images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(train_labels), 784)

    with open(test_labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        test_labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(test_images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        test_images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(test_labels), 784)

    return train_images.T, train_labels, test_images.T, test_labels


def onehot(label):
    e = np.zeros((10, 1))
    e[label] = 1.0
    return e


def load_dataset(path="./MNIST/raw"):
    train_images, train_label_lists, test_images, test_label_lists = load_mnist(path)
    train_labels = []
    test_labels = []
    for i in train_label_lists:
        train_labels.append(onehot(train_label_lists[i]))
    for i in test_label_lists:
        test_labels.append(onehot(test_label_lists[i]))
    return train_images.T, np.array(train_labels).squeeze(), test_images.T, np.array(test_labels).squeeze()


def show():
    train_images, train_labels, test_images, test_labels = load_mnist()
    print(train_images.shape)
    img = train_images[0, :].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.show()
