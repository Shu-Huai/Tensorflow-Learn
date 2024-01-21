import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import History
from keras.datasets import mnist
from keras.layers import *
from keras.models import Sequential
from numpy import ndarray
from tensorflow import keras

from config import *
from utils import plot


def check_device():
    print("设备：" + str(tf.config.list_physical_devices('GPU')))


def load_data() -> tuple[tuple[ndarray, ndarray], tuple[ndarray, ndarray]]:
    return mnist.load_data(path=os.path.join(os.path.dirname(__file__), data_path))


def preprocess(x_train: ndarray, x_test: ndarray) -> tuple[ndarray, ndarray]:
    x_train = x_train.reshape(60000, 28, 28, 1) / 255.0
    x_test = x_test.reshape(10000, 28, 28, 1) / 255.0
    return x_train, x_test


def hot_encoding(y_train: ndarray, y_test: ndarray) -> tuple[ndarray, ndarray]:
    y_train_vector = keras.utils.to_categorical(y_train)
    y_test_vector = keras.utils.to_categorical(y_test)
    return y_train_vector, y_test_vector


def show_images(x_train: ndarray, y_train: ndarray, save: bool = True) -> None:
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
    model = Sequential([
        Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    compile_model(model)
    return model


def compile_model(model: Sequential) -> None:
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.SGD(learning_rate=0.01),
                  metrics=['accuracy'])


def fit(model: Sequential, x_train: ndarray, y_train: ndarray, x_test: ndarray, y_test: ndarray) -> History:
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        validation_data=(x_test, y_test))
    return history


def show_indexes(history: History, save: bool = True) -> None:
    plot(save, os.path.join(os.path.dirname(__file__), plot_path, 'loss.png') if save else None,
         train_loss=history.history['loss'],
         validation_loss=history.history['val_loss'])
    plot(save, os.path.join(os.path.dirname(__file__), plot_path, 'loss.png') if save else None,
         train_acuracy=history.history['accuracy'],
         validation_acurac=history.history['val_accuracy'])


def predict(model: Sequential, x_test: ndarray) -> ndarray:
    return model.predict(x_test)


def score(model: Sequential, x_test: ndarray, y_test: ndarray) -> float:
    return model.evaluate(x_test, y_test, verbose=1)[1]


def save_model(model: Sequential) -> None:
    model.save(model_path)


def show_accuracy(predict_value: ndarray, test_value: ndarray, save: bool = True) -> None:
    plot(save, os.path.join(os.path.dirname(__file__), plot_path, 'accuracy.png') if save else None, 4, 10,
         predict_value=predict_value,
         test_value=test_value,
         difference=test_value - predict_value)


def load_model() -> Sequential:
    return keras.models.load_model(model_path)


def run(save: bool = True) -> None:
    check_device()
    (x_train, y_train), (x_test, y_test) = load_data()
    print("训练集x：" + str(x_train.shape))
    print("训练集y：" + str(y_train.shape))
    show_images(x_train, y_train, save)
    x_train, x_test = preprocess(x_train, x_test)
    y_train_vector, y_test_vector = hot_encoding(y_train, y_test)
    model: Sequential = build_network()
    history = fit(model, x_train, y_train_vector, x_test, y_test_vector)
    show_indexes(history, save)
    result: ndarray = predict(model, x_test)
    print("模型分数：" + str(score(model, x_test, y_test_vector)))
    show_accuracy(np.argmax(result[:100], axis=-1), y_test[:100], save)
    if save:
        save_model(model)
    model: Sequential = load_model()
    model.summary()
    test(model)


def test(model: Sequential) -> None:
    for item in os.listdir(draw_path):
        img = (cv2.imdecode(np.fromfile(os.path.join(draw_path, item), dtype=np.uint8), 0)
               .reshape(1, 28, 28, 1) / 255.0)
        print("图片：" + item + "，预测：" + str(np.argmax(model.predict(img))))


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = plot_dpi
    run(False)
