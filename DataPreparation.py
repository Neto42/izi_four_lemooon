import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from tensorflow.keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix

import cv2
import os

import numpy as np


class DataPreparation():
    def __init__(self):
        self.labels = ['ru', 'xz']
        self.img_size = 224

    def get_data(self, data_dir):
        data = []
        for label in self.labels:
            path = os.path.join(data_dir, label)
            class_num = self.labels.index(label)
            for img in os.listdir(path):
                try:
                    img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]
                    resized_arr = cv2.resize(img_arr, (self.img_size, self.img_size))
                    data.append([resized_arr, class_num])
                except Exception as e:
                    print(f'Невалидное изображение {path}/{img}')

        return np.array(data, dtype=object)

    def graph(self):
        print(f'[INFO] Подготовка данных...')

        train = self.get_data('state_number2/train')
        val = self.get_data('state_number2/test')

        data = []
        for i in train:
            if i[1] == 0:
                data.append("ru")
            else:
                data.append("car")
        sns.set_style('darkgrid')
        plot = sns.countplot(x=data)
        plot.figure.savefig("output/diagram.png")

        plt.figure(figsize=(5, 5))
        plot = plt.imshow(train[1][0])
        plt.title(self.labels[train[0][1]])
        plot.figure.savefig("output/ru.png")

        plt.figure(figsize=(5, 5))
        plot = plt.imshow(train[-1][0])
        plt.title(self.labels[train[-1][1]])
        plot.figure.savefig("output/xz.png")

        return train, val

    def data_preprocessing(self):
        train, val = self.graph()

        x_train = []
        y_train = []
        x_val = []
        y_val = []

        for feature, label in train:
            x_train.append(feature)
            y_train.append(label)

        for feature, label in val:
            x_val.append(feature)
            y_val.append(label)

        x_train = np.array(x_train) / 255
        x_val = np.array(x_val) / 255

        x_train.reshape(-1, self.img_size, self.img_size, 1)
        y_train = np.array(y_train)

        x_val.reshape(-1, self.img_size, self.img_size, 1)
        y_val = np.array(y_val)

        data_gen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=30,
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False)

        data_gen.fit(x_train)

        return x_train, x_val, y_train, y_val, data_gen

    def define_the_model(self):
        x_train, x_val, y_train, y_val, data_gen = self.data_preprocessing()

        model = Sequential()
        model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(224, 224, 3)))
        model.add(MaxPool2D())

        model.add(Conv2D(32, 3, padding="same", activation="relu"))
        model.add(MaxPool2D())

        model.add(Conv2D(64, 3, padding="same", activation="relu"))
        model.add(MaxPool2D())
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(2, kernel_initializer='uniform', activation='relu'))

        model.summary()

        opt = Adam(learning_rate=0.000001)
        model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

        return history, model, x_val, y_val

    def evaluating_the_result(self):
        history, model, x_val, y_val = self.define_the_model()

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(10)

        plt.figure(figsize=(15, 15))
        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

        print("[INFO] оценка сети...")
        predictions = model.predict(x_val, batch_size=32)
        print(f'{predictions}')
        scores = model.evaluate(x_val, y_val, verbose=0)
        print(f'Точность: {scores[1] * 100}')
        print(classification_report(y_val, predictions.argmax(axis=1), target_names=['ru (Class 0)', 'car (Class 1)']))
        # print(confusion_matrix(y_val, predictions.argmax(axis=1)))


if __name__ == '__main__':
    import time

    start_time = time.time()
    DataPreparation().evaluating_the_result()
    print(f'Время: {time.time() - start_time}')
