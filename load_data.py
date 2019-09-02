import matplotlib.pyplot as plt

import cv2
import numpy as np
from keras.utils import to_categorical


# 存在RGB图像，转为灰度
def read_line(line):
    filename = line[0:-6]
    label = int(line[-2])
    im = plt.imread(filename)

    I = cv2.resize(im, dsize=(96, 96))
    if I.shape == (96, 96, 3):
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    return I, label


# 输入：(txt)图片路径，label
# 输出：(numpy, numpy) 96x96x(num)的图片, one-hot label
def load_one_data(filename):

    print('Loading '+filename+'...')
    with open(filename, 'r') as f:
        lines = f.readlines()

    I, label = read_line(lines[0])
    labels = []
    labels.append(label)

    for i in range(1, len(lines)):
        J, label = read_line(lines[i])
        I = np.dstack((I, J))
        labels.append(label)

    labels = to_categorical(np.array(labels))


    return I, labels


# 输入：(int) 测试folder，txt文件的根目录
# 输出：(numpy) test_set, train_set
#
#   test_set = (X, label)
#             X     大小：测试集里图片的数量x96x96；数据类型float32；取值范围(0, 1)
#             label 大小：测试集里图片的数量x8；one-hot编码
#
#   train_set = (X, label)
#             X     大小：训练集里图片的数量x96x96；数据类型float32；取值范围(0, 1)
#             label 大小：训练集里图片的数量x8；one-hot编码
#
def load_data(i, root='./10fold'):
    print('Test Set is ' + str(i) + ' folder...')

    test_set_filename = root+'/'+str(i)+'.txt'
    test_set_x, test_set_y = load_one_data(test_set_filename)
    test_set_x = test_set_x.transpose((2, 0, 1))
    test_set = (test_set_x, test_set_y)

    if i == 0:
        filename = root + '/' + str(1) + '.txt'
        X, Y = load_one_data(filename)
        for j in range(2, 10):
            filename = root + '/' + str(j) + '.txt'
            X1, Y1 = load_one_data(filename)
            X = np.concatenate((X, X1), axis=2)
            Y = np.concatenate((Y, Y1), axis=0)

    else:
        filename = root + '/' + str(0) + '.txt'
        X, Y = load_one_data(filename)
        for j in range(1, 10):
            if j != i:
                filename = root+'/'+str(j)+'.txt'
                X1, Y1 = load_one_data(filename)
                X = np.concatenate((X, X1), axis=2)
                Y = np.concatenate((Y, Y1), axis=0)

    X = X.transpose((2, 0, 1))
    train_set = (X, Y)
    print('Data Loaded')

    return train_set, test_set


if __name__ == '__main__':

    train_set, test_set = load_data(2)

    X, Label = train_set

    print(X.shape)
    print(Label.shape)

    for i in range(10):
        I = X[i, :, :]
        label = Label[i, :]
        plt.imshow(I, cmap='gray')
        plt.title(str(np.argmax(label)))
        plt.show()
