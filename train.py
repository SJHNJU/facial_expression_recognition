from model import *
from load_data import *

from keras.models import load_model

from keras.datasets import mnist

# 网络：model.py里get_mnist_model的模型
# 输出：训练准确率，model.h5模型文件
# 用于验证代码正确性使用
def train_mnist():
    print('loading data...')

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    img_x, img_y = 28, 28
    x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
    x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print(x_train.shape)
    print(y_train.shape)

    print('Loading model...')
    model = get_mnist_model(28, 28)

    model = load_model('model.h5')

    print('Begin training...')
    model.fit(x_train, y_train, batch_size=64, epochs=10)

    model.save('model.h5')

    score = model.evaluate(x_test, y_test)
    print('acc', score[1])


def train():
    print('loading data...')
    (x_train, y_train), (x_test, y_test) = load_data(0)

    img_x, img_y = 96, 96
    x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
    x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)

    print(x_train.shape)
    print(y_train.shape)

    print('Loading model...')
    model = get_model(96, 96)

    print('Begin training...')
    model.fit(x_train, y_train, batch_size=64, epochs=5)

    model.save('model.h5')

    score = model.evaluate(x_test, y_test)
    print('acc', score[1])


def predict(modelname, string):
    print('Loading model...')
    model = load_model(modelname)
    print('model Loaded')

    line = string
    filename = line[0:-5]
    label = int(line[-1])
    im = plt.imread(filename)

    I = cv2.resize(im, dsize=(96, 96))
    if I.shape == (96, 96, 3):
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    plt.imshow(I, cmap='gray')
    plt.show()
    print('Label is '+ str(label))

    I = I.reshape((1, 96, 96, 1))
    predicted = model.predict(I)
    print('Predict Vector is {0}'.format(predicted))
    print('Predict category is {0}'.format(np.argmax(predicted)))


if __name__ == '__main__':

    # train()
    predict('model.h5', './CK+/cohn-kanade-images/S108/008/S108_008_00000011.png    5')
