# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input


def do_predict(model_h5_list, categories, img_width, img_height, input_folder, output_file):
    image.LOAD_TRUNCATED_IMAGES = True

    model_list = []
    for model_h5 in model_h5_list:
        model_list.append(load_model(
            model_h5,
            compile=False))

    rows = []
    column_names = ['id', 'category']
    # pic_list = os.listdir(input_folder)
    for f in range(440):
        f = str(f) + '.jpg'
        if not f.startswith('.'):
            try:
                img = image.load_img(input_folder + '/' + f, target_size=(img_width, img_height))
                img_array = image.img_to_array(img)
                x = np.expand_dims(img_array, axis=0)
                x_orig = preprocess_input(x)

                y_prob = np.zeros([1, 12])

                for model in model_list:
                    x1 = x_orig
                    y_prob1 = model.predict(x1)
                    x2 = np.rot90(x_orig, k=1, axes=(1, 2))
                    y_prob2 = model.predict(x2)
                    x3 = np.rot90(x_orig, k=2, axes=(1, 2))
                    y_prob3 = model.predict(x3)
                    x4 = np.rot90(x_orig, k=3, axes=(1, 2))
                    y_prob4 = model.predict(x4)

                    y_prob += (y_prob1 + y_prob2 + y_prob3 + y_prob4) / 4.

                # y_prob = model.predict(x)
                y_classes = y_prob.argmax(axis=-1)
                cat_id = y_classes[0]

                row = [str(f), str(categories[cat_id])]
                rows.append(row)
                print(row)
            except Exception as e:
                print(e.message)
                print('Canot predict image: ' + f)

    df = pd.DataFrame(rows, columns=column_names)
    df.to_csv(output_file, index=False, header=False)
    print('Done')


if __name__ == "__main__":
    model_path_list = []
    model_path_list.append('D:/Datasets/GuangDong/output-inc-res-90/model_80_0.04.h5')
    model_path_list.append('D:/Datasets/GuangDong/output-NASNET/model_35_0.04.h5')
    model_path_list.append('D:/Datasets/GuangDong/output 2018 10 02 Focal loss/model_20_0.05.h5')
    model_path_list.append('D:/Datasets/GuangDong/output tf keras 20181004/model_80_0.00.h5')
    metadata_path = os.path.join('C:/Users/AlphaCat/Desktop/models/gmle_cnn_keras_classifier_resnet50/cnn_model',
                                 'metadata.txt')
    test_path = os.path.join('D:/Datasets/GuangDong/guangdong_round1_test_a_20180916')

    meta_file = open(metadata_path, 'rb')
    categories = [str(i.strip().decode('UTF-8')) for i in meta_file.readlines()]

    # categories = ['defect', 'norm']

    do_predict(model_path_list, categories, 299, 299, test_path, 'output.csv')
