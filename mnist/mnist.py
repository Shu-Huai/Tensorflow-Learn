import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import History
from keras.datasets import mnist
from keras.layers import *
from keras.models import Sequential
from numpy import ndarray
from tensorflow import keras

from config import *
from utils import plot, Param, check_device


def load_data() -> tuple[tuple[ndarray, ndarray], tuple[ndarray, ndarray]]:
    """
    加载数据
    :return: 无
    """
    return mnist.load_data(path=os.path.join(os.path.dirname(__file__), data_path))


def preprocess(x_train: ndarray, x_test: ndarray) -> tuple[ndarray, ndarray]:
    """
    归一化处理
    :param x_train:待处理的数据
    :param x_test:待处理的数据
    :return:处理后的数据
    """
    x_train = x_train.reshape(60000, 28, 28, 1) / 255.0
    x_test = x_test.reshape(10000, 28, 28, 1) / 255.0
    return x_train, x_test


def hot_encoding(y_train: ndarray, y_test: ndarray) -> tuple[ndarray, ndarray]:
    """
    便签热编码
    将便签集进行one-hot编码分类，每个类别重新转化为一个二进制向量，以便更好训练网络模型
    :param y_train:待处理的数据
    :param y_test:待处理的数据
    :return:
    """
    y_train_vector = keras.utils.to_categorical(y_train)
    y_test_vector = keras.utils.to_categorical(y_test)
    return y_train_vector, y_test_vector


def show_images(x_train: ndarray, y_train: ndarray, save: bool = True) -> None:
    """
    画出数据集的样子
    :param x_train:数据集x
    :param y_train:数据集y
    :param save:是否保存结果
    :return:无
    """
    plt.figure(figsize=(10, 2))
    for i, image in enumerate(x_train[:10]):
        plt.subplot(1, 10, i + 1)
        plt.imshow(image, cmap="binary")
        plt.axis('off')
    plt.text(1, 1, str(y_train[:10]))
    if save:
        plt.savefig(r".\plots\images.png")
    plt.show()


def build_network() -> Sequential:
    """
    构架神经网络
    :return:
    """
    model = Sequential([
        # 卷积层：filter=32表示经过这层卷积要得到特征图数量32，kernel_size=3卷积核尺寸为3x3等同于(3,3)，使用relu激活函数，输入大小为28*28*3
        Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
        # 池化层：使用最大池化，一般参数不改，一般使W,H变为原来的1/2
        MaxPooling2D((2, 2)),
        Conv2D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling2D((2, 2)),
        # 将卷积后的立体特征图拉长一条数据，变成一条向量，为为全连接层准备
        Flatten(),
        Dense(64, activation='relu'),
        # 得到最后十个类别，使用交叉熵
        Dense(10, activation='softmax')
    ])
    compile_model(model)
    return model


def compile_model(model: Sequential) -> None:
    """
    编译模型
    :param model:模型实例
    :return:无
    """
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.SGD(learning_rate=0.01),
                  metrics=['accuracy'])


def fit(model: Sequential, x_train: ndarray, y_train: ndarray, x_test: ndarray, y_test: ndarray) -> History:
    """
    训练模型
    :param model:模型实例
    :param x_train:训练集x
    :param y_train:训练集y
    :param x_test:测试集x
    :param y_test:测试集y
    :return:History对象
    """
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        validation_data=(x_test, y_test))
    return history


def show_indexes(history: History, save: bool = True) -> None:
    """
    画出loss和准确率函数
    :param history: History实例
    :param save: 是否保存图片
    :return: 无
    """
    plot(save, os.path.join(os.path.dirname(__file__), plot_path, 'loss.png') if save else None,
         train_loss=history.history['loss'],
         validation_loss=history.history['val_loss'])
    print(f"min_loss: {min(history.history['loss'])}")
    plot(save, os.path.join(os.path.dirname(__file__), plot_path, 'loss.png') if save else None,
         train_acuracy=history.history['accuracy'],
         validation_acurac=history.history['val_accuracy'])
    print(f"accuracy: {max(history.history['accuracy'])}")


def predict(model: Sequential, x_test: ndarray) -> ndarray:
    """
    预测
    :param model: 模型实例
    :param x_test: 测试x
    :return: 预测结果
    """
    return model.predict(x_test)


def score(model: Sequential, x_test: ndarray, y_test: ndarray) -> float:
    """
    评估模型分数
    :param model: 模型实例
    :param x_test: 测试x
    :param y_test: 测试y
    :return: 模型分数
    """
    return model.evaluate(x_test, y_test, verbose=1)[1]


def save_model(model: Sequential) -> None:
    """
    将模型保存到文件
    :param model: 模型实例
    :return: 无
    """
    model.save(model_path)


def show_accuracy(predict_value: ndarray, test_value: ndarray, save: bool = True) -> None:
    """
    画出预测值、实际值即偏差
    :param predict_value: 预测是
    :param test_value: 真实值
    :param save: 是否保存结果
    :return: 无
    """
    plot(save, os.path.join(os.path.dirname(__file__), plot_path, 'accuracy.png') if save else None, 4, 10,
         predict_value=Param(predict_value, label="predict_value", marker="o"),
         test_value=Param(test_value, label="test_value", marker="*", line_style="-"),
         difference=test_value - predict_value)


def load_model() -> Sequential:
    """
    从文件加载模型
    :return: 模型实例
    """
    return keras.models.load_model(model_path)


def run(save: bool = True) -> None:
    """
    启动！！
    :param save: 是否保存结果
    :return: 无
    """
    check_device()
    (x_train, y_train), (x_test, y_test) = load_data()
    print(f"训练集x：{str(x_train.shape)}")
    print(f"训练集y：{str(y_train.shape)}")
    show_images(x_train, y_train, save)
    x_train, x_test = preprocess(x_train, x_test)
    y_train_vector, y_test_vector = hot_encoding(y_train, y_test)
    model: Sequential = build_network()
    history = fit(model, x_train, y_train_vector, x_test, y_test_vector)
    show_indexes(history, save)
    result: ndarray = predict(model, x_test)
    print(f"模型分数：{str(score(model, x_test, y_test_vector))}")
    show_accuracy(np.argmax(result[:100], axis=-1), y_test[:100], save)
    if save:
        save_model(model)
    model: Sequential = load_model()
    model.summary()
    test(model)


def test(model: Sequential) -> None:
    """
    用手写的png测试模型
    :param model: 模型实例
    :return: 无
    """
    for item in os.listdir(draw_path):
        img = (cv2.imdecode(np.fromfile(os.path.join(draw_path, item), dtype=np.uint8), 0)
               .reshape(1, 28, 28, 1) / 255.0)
        print(f"图片：{item}，预测：{str(np.argmax(model.predict(img)))}")


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = plot_dpi
    run(False)
