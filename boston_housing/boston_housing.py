import numpy as np
import tensorflow.keras as keras
from keras.callbacks import History
from keras.datasets import boston_housing
import tensorflow as tf
from keras.layers import *
from keras.models import Sequential
from numpy import ndarray
from sklearn.metrics import r2_score

from config import *
from utils import *


def load_data() -> tuple[tuple[ndarray, ndarray], tuple[ndarray, ndarray]]:
    return boston_housing.load_data(path=os.path.join(os.path.dirname(__file__), data_path))


def check_device():
    print("设备：" + str(tf.config.list_physical_devices('GPU')))


def preprocess(x_train: ndarray, x_test: ndarray) -> tuple[ndarray, ndarray]:
    mean = x_train.mean(axis=0)
    x_train -= mean
    std = x_train.std(axis=0)
    x_train /= std
    x_test -= mean
    x_test /= std
    return x_train, x_test


def build_network(shape: int) -> Sequential:
    model: Sequential = Sequential([
        Dense(64, activation='relu', input_shape=(shape,)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    compile_model(model)
    return model


def compile_model(model: Sequential) -> None:
    model.compile(optimizer=keras.optimizers.RMSprop(0.01),
                  loss=keras.losses.mse,
                  metrics=keras.metrics.mae)


def fit(model: Sequential, x_train: ndarray, y_train: ndarray, x_test: ndarray, y_test: ndarray) -> History:
    history: History = model.fit(x_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=1,
                                 validation_data=(x_test, y_test))
    return history


def show_mae(history: History, save: bool = True) -> None:
    plot(save, os.path.join(os.path.dirname(__file__), plot_path, 'mean_absolute_error.png') if save else None,
         mean_absolute_error=history.history['mean_absolute_error'])


def predict(model: Sequential, x_test: ndarray) -> ndarray:
    return model.predict(x_test, verbose=1)


def score(predict_value: ndarray, test_value: ndarray) -> float:
    return r2_score(test_value, predict_value)


def show_accuracy(predict_value: ndarray, test_value: ndarray, save: bool = True) -> None:
    plot(save, os.path.join(os.path.dirname(__file__), plot_path, 'accuracy.png') if save else None,
         test_value=test_value,
         predict_value=predict_value,
         difference=test_value - predict_value)


def save_model(model: Sequential) -> None:
    model.save(model_path)


def load_model() -> Sequential:
    return keras.models.load_model(model_path)


def run(save: bool = True) -> None:
    (x_train, y_train), (x_test, y_test) = load_data()
    print("训练集x：" + str(x_train.shape))
    print("训练集y：" + str(y_train.shape))
    x_train, x_test = preprocess(x_train, x_test)
    model: Sequential = build_network(x_train.shape[1])
    history = fit(model, x_train, y_train, x_test, y_test)
    show_mae(history)
    result: ndarray = predict(model, x_test)
    print("模型分数：" + str(score(result, y_test)))
    show_accuracy(np.array([item[0] for item in result]), y_test)
    if save:
        save_model(model)
    model: Sequential = load_model()
    model.summary()


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = plot_dpi
    run(False)
