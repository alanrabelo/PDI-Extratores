import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder



class DataManager:

    def loadData(self):

        file_url = 'ocr_car_numbers_rotulado.txt'

        X = []
        y = []

        with open(file_url, 'r') as f:
            for line in f.readlines():
                new_line = line.replace(' ', ',').replace('\n', '').split(',')
                label = new_line[-1]
                data =  np.reshape(np.array(new_line[:-1], dtype=float), newshape=(35, 35))
                X.append(data)
                y.append(label)

        encoder = LabelEncoder()
        encoder.fit(y)

        return X, encoder.transform(y), encoder
