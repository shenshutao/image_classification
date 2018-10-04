# -*- coding: utf-8 -*-
import os

import numpy as np
import tensorflow as tf  # only work from tensorflow==1.9.0-rc1 and after
from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from skimage.util import random_noise
import cv2
from tensorflow.python.summary import summary as tf_summary

_EPOCHS = 80
_BATCH_SIZE = 4
_IMAGE_SIZE = (512, 512)
_NUM_CLASS = 12

_CLASSES = ['norm', 'defect1', 'defect2', 'defect3', 'defect4', 'defect5', 'defect6', 'defect7', 'defect8', 'defect9',
            'defect10', 'defect11']


# tf.enable_eager_execution()

def main(_):
    train_tfdata, train_no = tfdata_from_dir('D:/Datasets/GuangDong/train', classes=_CLASSES)
    print('{} images founded'.format(train_no))
    #    test_tfdata, test_no = tfdata_from_dir('D:/Datasets/GuangDong/train_try')
    #
    # iterator = train_tfdata.make_one_shot_iterator()
    # next_element = iterator.get_next()
    #
    # with tf.Session() as sess:
    #     value = sess.run(next_element)
    #     print(value)
    # print(value[0][0].shape)
    # img = tf.keras.preprocessing.image.array_to_img(value[0][0])
    # img.save('aug.jpg')
    # exit(-1)

    # Get Model
    model = keras_model()  # your keras model here
    model.summary()

    tensor_board = TensorBoard(log_dir='log', histogram_freq=0, write_graph=True, write_grads=True, write_images=True)
    save_every_5 = ModelCheckpoint(filepath='output/model_{epoch:02d}_{loss:.2f}.h5', verbose=1, save_best_only=False,
                                   save_weights_only=False, mode='auto', period=5)
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

    model.compile(SGD(lr=0.001, momentum=0.9), focal_loss_category(), metrics=['acc'])
    model.fit(
        train_tfdata.make_one_shot_iterator(),
        steps_per_epoch=int(train_no / _BATCH_SIZE),
        epochs=_EPOCHS,
        #   validation_data=test_tfdata.make_one_shot_iterator(),
        #  validation_steps=int(test_no / _BATCH_SIZE),
        callbacks=[tensor_board, learning_rate_reduction, save_every_5],
        class_weight='auto',
        verbose=1)


def tfdata_from_dir(directory, classes=None):
    if classes is None:
        classes = []
        for subdir in sorted(os.listdir(directory)):
            if os.path.isdir(os.path.join(directory, subdir)):
                classes.append(subdir)
    class_indices = dict(zip(classes, range(len(classes))))

    img_list_x = []
    label_list_y = []
    for subdir, label in class_indices.items():
        print('class: {}, label: {}, number:{}'.format(subdir, label, len(os.listdir(os.path.join(directory, subdir)))))
        for img in os.listdir(os.path.join(directory, subdir)):
            img_list_x.append(os.path.join(directory, subdir, img))
            label_list_y.append(label)

    num_class = len(classes)

    return get_img_dataset(img_list_x, label_list_y, num_class, True), len(img_list_x)


def get_img_dataset(img_list_x, label_list_y, num_class, aug=True):
    def _parse_function(img_path, label, img_size):
        image_string = tf.read_file(img_path)
        image = tf.image.decode_image(image_string)
        image.set_shape([None, None, 3])
        image = tf.image.resize_images(image, [img_size[0], img_size[1]])

        label_one_hot = tf.one_hot(tf.cast(label, tf.uint8), num_class)
        return image, label_one_hot

    def augment(x, y):
        # 水平翻转 垂直翻转
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)

        # 随机对角线翻转
        if np.random.rand() > 0.5:
            x = tf.image.transpose_image(x)

        # 随机亮度
        x = tf.image.random_brightness(x, max_delta=0.1)

        # 随机裁剪
        size = _IMAGE_SIZE[0]
        s = np.random.randint(0, x.shape[1] - size)
        x = x[s:s + size, s:s + size]

        # 随机噪声
        print(type(x))
        # x = random_noise(x, mode='gaussian', clip=True) * 255
        # img = tf.keras.preprocessing.image.array_to_img(x[0])
        # img.save('aug.jpg')
        # summary1 = tf_summary.image('augment', x)
        # writer = tf_summary.FileWriter('.log')
        # writer.add_summary(summary1)

        return x, y

    def preprocess(x, y):
        x = preprocess_input(x)
        return x, y

    dataset = tf.data.Dataset.from_tensor_slices((img_list_x, label_list_y))
    dataset = dataset.shuffle(1000)
    dataset = dataset.map(lambda x, y: _parse_function(x, y, [_IMAGE_SIZE[0] + 20, _IMAGE_SIZE[1] + 20]))
    if aug:
        dataset = dataset.map(lambda x, y: augment(x, y))
    dataset = dataset.map(lambda x, y: preprocess(x, y))
    dataset = dataset.repeat(_EPOCHS)
    dataset = dataset.batch(_BATCH_SIZE)

    return dataset


def keras_model():
    # don't include the top (final FC) layers.
    base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(_IMAGE_SIZE + (3,)))

    # add FC layers.
    x = base_model.output
    x = Dropout(0.5, name='dropout_1')(x)
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = Dense(_NUM_CLASS, activation='softmax', name='output_layer')(x)

    # this is the final model we will train
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    return model


def focal_loss_category(gamma=2.):
    def focal_loss(y_true, y_pred):
        return - tf.reduce_sum(y_true * ((1 - y_pred) ** gamma) * tf.log(y_pred), axis=1)

    return focal_loss


if __name__ == '__main__':
    tf.app.run()
