import cv2
import os

import numpy as np
from sklearn.metrics import classification_report


class DataPrediction():
    def __init__(self):
        self.str1 = 'frame'
        self.str2 = 'logo'
        self.labels = [self.str1]
        self.img_size = 224
        self.path_img = 'frame'

    def get_data(self, data_dir):
        data = []
        images = []

        for label in self.labels:
            path = os.path.join(data_dir, label)
            class_num = self.labels.index(label)
            for img in os.listdir(path):
                # print(f'Изображение: {img}')
                images.append(img)
                try:
                    img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]
                    resized_arr = cv2.resize(img_arr, (self.img_size, self.img_size))
                    data.append([resized_arr, class_num])
                except Exception as e:
                    print(f'Невалидное изображение {path}/{img}')

        return np.array(data, dtype=object), images

    def data_prediction(self):
        import time

        start_time = time.time()

        train, images = self.get_data(self.path_img)

        print(f'[INFO] Подготовка кадра...')

        x_train = []
        y_train = []

        for feature, label in train:
            x_train.append(feature)
            y_train.append(label)

        x_train = np.array(x_train) / 255
        x_train.reshape(-1, self.img_size, self.img_size, 1)

        x_train.reshape(-1, self.img_size, self.img_size, 1)
        y_train = np.array(y_train)

        print(f'[INFO] Загрузка обученой модели...')
        from tensorflow.python.keras.saving.save import load_model
        model = load_model(f'model_cnn/my_model1.tf')
        # model = load_model(f'model_cnn/{self.str1}_{self.str2}.tf')

        print("[INFO] Оценка сети...")
        predictions = model.predict_classes(x_train, batch_size=1)
        print(f'{predictions}')
        # print(classification_report(y_train, predictions))
        # print(confusion_matrix(y_val, predictions.argmax(axis=1)))
        scores = model.evaluate(x_train, y_train, verbose=0)
        print(f'Точность: {scores[1] * 100}')

        scores = model.evaluate(x_train, y_train, verbose=0)
        print(f'Базовая ошибка {100 - scores[1] * 100}')

        print(f'Время обучения: {time.time() - start_time}')

        return images, predictions


if __name__ == '__main__':
    DataPrediction().data_prediction()
