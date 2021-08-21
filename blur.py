from PIL import Image, ImageFilter
from prediction import DataPrediction
import pandas as pd


class Blur(DataPrediction):
    def blur(self):
        images, predictions = self.data_prediction()

        list1 = zip(images, predictions)

        for img, prediction in list1:
            try:
                if prediction == 1:
                    original_image = Image.open(f'{self.path_img}/{self.str1}/{img}')
                    blurred_image = original_image.filter(ImageFilter.GaussianBlur(radius=50))
                    blurred_image.save(f'{self.path_img}/blur/blur_{img}')
            except Exception as e:
                print(f'[INFO] Изображение {img} имеет неправильный режим ')

    # df = pd.DataFrame({
    #     'img': images,
    #     'prediction': predictions,
    #     'y': y_train
    # })
    # df.to_excel('output1/1.xlsx')


if __name__ == '__main__':
    import time

    start_time = time.time()
    Blur().blur()
    print(f'Время для блюра: {time.time() - start_time}')
