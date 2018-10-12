# -*- coding=utf-8 -*-
import os
import sys

import argparse
import datetime
import subprocess
import tensorflow as tf
import pandas as pd
from keras import backend as K
# from keras.applications.xception import Xception, preprocess_input
# from keras.applications.nasnet import NASNetLarge, preprocess_input
# from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard, Callback)
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from cnn_model import util
from cnn_model import dataset_download


def main():
    # For big image.
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    parsed_args = default_args(sys.argv[1:])
    input_path = parsed_args['input_path']
    remote_output_dir = parsed_args['output_path']
    project = parsed_args['project']
    job_name = parsed_args['job_name']
    param_epoch = int(parsed_args['param_epoch'])
    param_batch_size = int(parsed_args['param_batch_size'])
    param_img_resize = int(parsed_args['param_img_resize'])
    param_learning_rate = float(parsed_args['param_learning_rate'])
    param_validate_set_ratio = float(parsed_args['param_validate_set_ratio'])
    data_augmentation = dict(
        fill_mode='nearest'
    )
    if parsed_args['augmentation_rotation_range'] is not None:
        data_augmentation['rotation_range'] = int(parsed_args['augmentation_rotation_range'])
    if parsed_args['augmentation_zoom_range'] is not None:
        data_augmentation['zoom_range'] = int(parsed_args['augmentation_zoom_range'])
    if parsed_args['augmentation_horizontal_flip'] is not None:
        data_augmentation['horizontal_flip'] = bool(parsed_args['augmentation_horizontal_flip'] == 'True')
    if parsed_args['augmentation_vertical_flip'] is not None:
        data_augmentation['vertical_flip'] = bool(parsed_args['augmentation_vertical_flip'] == 'True')
    if parsed_args['augmentation_width_shift_range'] is not None:
        data_augmentation['width_shift_range'] = float(parsed_args['augmentation_width_shift_range']),
    if parsed_args['augmentation_height_shift_range'] is not None:
        data_augmentation['height_shift_range'] = bool(parsed_args['augmentation_height_shift_range'] == 'True')
    if parsed_args['augmentation_brightness_range'] is not None:
        data_augmentation['brightness_range'] = (1. - float(parsed_args['augmentation_brightness_range']),
                                                 1. + float(parsed_args['augmentation_brightness_range']))

    print(data_augmentation)

    print('======== Get Dataset =========')
    dataset_download.get_data(input_path, 'data/train_data')

    if not os.path.exists('output'):
        os.makedirs('output')

    print('======== Upload metadata ========')
    categories = os.listdir('data/train_data')
    with open('output/metadata.txt', 'w') as meta_file:
        meta_file.writelines(["%s\n" % item for item in categories])

    # upload to google bucket
    if remote_output_dir.startswith('gs://'):
        subprocess.check_call([
            'gsutil', '-m', '-q', 'cp', 'output/metadata.txt', os.path.join(remote_output_dir, 'metadata.txt'),
        ])

    print('======== Split data into train / validate sets =========')
    util.split_data('data/train_data', 'data/validate_data', param_validate_set_ratio)

    print('======== Train resNet Model !!! =========')
    train_resnet(remote_output_dir, categories, 'data/train_data', 'data/validate_data', 'output/last_model.h5',
                 data_augmentation, param_epoch, param_batch_size, param_img_resize, param_img_resize,
                 param_learning_rate)

    print('======== Convert H5 to SavedModel =========')
    util.convert_h5_to_pb('output', 'last_model.h5')

    print('======== Upload model to destination =======')
    if remote_output_dir.startswith('gs://'):
        subprocess.check_call([
            'gsutil', '-m', '-q', 'cp', '-r', 'output/*', remote_output_dir
        ])

    print('======== Job finished =======')


def default_args(argv):
    """Provides default values for Workflow flags."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True, help='Input specified as uri to ZIP file.')
    parser.add_argument('--output_path', required=True, help='Output directory to write results to.')
    parser.add_argument('--project', type=str, help='The cloud project name to be used for running this pipeline')
    parser.add_argument('--job_name', type=str, default='job-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
                        help='A unique job identifier.')

    parser.add_argument('--param_epoch', type=str, default='50', help='Deep learning epoch No.')
    parser.add_argument('--param_batch_size', type=str, default='16', help='Deep learning batch size.')
    parser.add_argument('--param_img_resize', type=str, default='299', help='Resize image to this pixel * pixel.')
    parser.add_argument('--param_validate_set_ratio', type=str, default='0.3', help='Deep learning validate set ratio')
    parser.add_argument('--param_learning_rate', type=str, default='0.0001', help='Augmentation learning rate.')

    parser.add_argument('--augmentation_rotation_range', type=str, default=None, help='Augmentation rotation range.')
    parser.add_argument('--augmentation_zoom_range', type=str, default=None, help='Augmentation rotation range.')
    parser.add_argument('--augmentation_horizontal_flip', type=str, default=None, help='Augmentation horizontal flip.')
    parser.add_argument('--augmentation_vertical_flip', type=str, default=None, help='Augmentation vertical flip.')
    parser.add_argument('--augmentation_width_shift_range', type=str, default=None, help='Aug width shift range.')
    parser.add_argument('--augmentation_height_shift_range', type=str, default=None, help='Aug height shift range.')
    parser.add_argument('--augmentation_brightness_range', type=str, default=None, help='Aug brightness range.')

    parsed_args, _ = parser.parse_known_args(argv)
    return vars(parsed_args)


def train_resnet(remote_output_dir, categories, train_data_dir, validate_data_dir, last_model_file_name,
                 data_augmentation, param_epoch=50, param_batch_size=32, img_width=299, img_height=299,
                 param_learning_rate=0.0001):

    categorie_size = len(categories)

    ####################
    # 1. Init data generator & augmentation
    ####################
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        **data_augmentation
    )
    validate_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        # save_to_dir='data/aug',
        target_size=(img_width, img_height),
        batch_size=param_batch_size,
        classes=categories,
        class_mode='categorical')
    validate_generator = validate_datagen.flow_from_directory(
        validate_data_dir,
        # save_to_dir='data/aug_test',
        target_size=(img_width, img_height),
        batch_size=param_batch_size,
        classes=categories,
        class_mode='categorical')

    ####################
    # 2. Init model structure
    ####################
    base_model, model = getKerasModel(categorie_size, img_height, img_width)

    ####################
    # 3. Init callbacks
    ####################
    tensor_board = TensorBoard(log_dir='log', histogram_freq=0, write_graph=True, write_grads=True, write_images=True)
    save_every_5 = ModelCheckpoint(monitor='loss', filepath=last_model_file_name, verbose=1, save_best_only=False,
                                   save_weights_only=False, mode='auto', period=5)
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=2, verbose=1, factor=0.5, min_lr=0.000001)
    upload_status_log = UploadStatusLog(output_path=remote_output_dir)

    #####################
    # 4. Transfer learning, max ( epoch / 2 , 2 ), accelerate training
    ####################
    # train only the FC layers (which were randomly initialized), freeze all convolutional resnet layers
    for layer in base_model.layers:
        layer.trainable = False
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='Adam', loss=focal_loss(), metrics=['accuracy'])
    # train the model on the new data for a few epochs
    model.fit_generator(
        train_generator,
        epochs=min(param_epoch / 2, 2),
        class_weight='auto',
        steps_per_epoch=train_generator.n // param_batch_size,  # Loop whole image set once per epoch
        validation_data=validate_generator,
        validation_steps=validate_generator.n // param_batch_size,
        callbacks=[upload_status_log, tensor_board, learning_rate_reduction, save_every_5])

    #####################
    # 5. Fine tuning
    ####################
    for layer in model.layers:
        layer.trainable = True
    # fine tune: stochastic gradient descent optimizer
    model.compile(optimizer=Adam(lr=param_learning_rate), loss=focal_loss(), metrics=['accuracy'])
    # fine tune: train again for fine tune
    model.fit_generator(
        train_generator,
        initial_epoch=min(param_epoch / 2, 2),
        epochs=param_epoch,
        class_weight='auto',
        steps_per_epoch=train_generator.n // param_batch_size,
        validation_data=validate_generator,
        validation_steps=validate_generator.n // param_batch_size,
        callbacks=[upload_status_log, tensor_board, learning_rate_reduction, save_every_5])

    model.save(last_model_file_name)


def getKerasModel(categorie_size, img_height, img_width):
    # # Base model Conv layers + Customize FC layers
    if K.image_data_format() == 'channels_first':
        the_input_shape = (3, img_width, img_height)
    else:
        the_input_shape = (img_width, img_height, 3)
    # don't include the top (final FC) layers.
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=the_input_shape)
    # add FC layers.
    x = base_model.output
    x = Dropout(0.5, name='dropout_1')(x)
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = Dense(categorie_size, activation='softmax', name='output_layer')(x)
    # this is the final model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()

    return base_model, model


class UploadStatusLog(Callback):
    """
    Upload real time status (csv file) to the remote output path.
    """
    line = 0

    def __init__(self, output_path):
        super().__init__()
        self.perform_list = []
        self.output_path = output_path
        self.epoch_val_acc = None
        self.epoch_val_loss = None

    def on_epoch_end(self, epoch, logs={}):
        print('on_epoch_end')
        UploadStatusLog.line += 1
        li = [UploadStatusLog.line, str(logs.get('loss')), str(logs.get('acc')), str(logs.get('val_loss')),
              str(logs.get('val_acc'))]
        self.perform_list.append(li)
        self.store_to_output()

    def store_to_output(self):
        df = pd.DataFrame(data=self.perform_list, columns=['index', 'loss', 'acc', 'val_loss', 'val_acc'])
        df.to_csv('output/realtime_status.csv', index=False)

        # upload to google bucket
        if self.output_path.startswith('gs://'):
            subprocess.check_call([
                'gsutil', '-m', '-q', 'cp', 'output/realtime_status.csv',
                os.path.join(self.output_path, 'realtime_status.csv'),
            ])

        print('Save Status File Successful !!')


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return - K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


if __name__ == "__main__":
    main()
