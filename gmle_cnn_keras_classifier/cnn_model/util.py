import os
import shutil
import zipfile

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from tensorflow.python.framework import graph_util

import pandas as pd
from keras.callbacks import Callback
import subprocess


def convert_h5_to_pb(dir, filename):
    model = load_model(os.path.join(dir, filename), compile=False)
    name = 'saved_model.pb'

    # Function 1
    output_names = [out.op.name for out in model.outputs]
    # Freezes the state of a session into a pruned computation graph.
    graph = K.get_session().graph
    with graph.as_default():
        input_graph_def = graph.as_graph_def()
        for node in input_graph_def.node:
            node.device = ""
        frozen_graph = graph_util.convert_variables_to_constants(K.get_session(), input_graph_def,
                                                                 output_names)
    with tf.gfile.GFile(os.path.join(dir, name), "wb") as f:
        f.write(frozen_graph.SerializeToString())

    # Function 2
    export_path = os.path.join(dir, 'saved_model2')
    with K.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            export_path,
            inputs={'input_image': model.input},
            outputs={t.name: t for t in model.outputs}
        )


def extract_zipfile(zip_file, target_folder):
    '''
    Extract zip file and remove system files
    '''
    zip_ref = zipfile.ZipFile(zip_file, 'r')
    zip_ref.extractall(target_folder)
    zip_ref.close()

    folders = os.listdir(target_folder)
    for fold in folders:
        if fold == '__MACOSX' or fold.startswith('.'):
            shutil.rmtree(os.path.join(target_folder, fold))

    for arg, dirnam, names in os.walk(target_folder):
        if '.DS_Store' in names:
            os.remove(os.path.join(arg, '.DS_Store'))


def split_data(train_data_dir, validate_data_dir, validate_precentage: float):
    '''
    Split data into train / validate set base on the ratio
    '''
    if not os.path.exists(validate_data_dir):
        os.makedirs(validate_data_dir)

    # split train / validate
    folders = os.listdir(train_data_dir)
    for fod in folders:
        if not fod.startswith('.'):
            files = os.listdir(os.path.join(train_data_dir, fod))
            if not os.path.exists(os.path.join(validate_data_dir, fod)):
                os.makedirs(os.path.join(validate_data_dir, fod))

            # Shuffle image
            np.random.shuffle(files)
            val_num = int(len(files) * float(validate_precentage))

            print('{} of {} in category {} move to validate set'.format(val_num, len(files), fod))
            print('Train :' + str(files[val_num:]))
            print('Validate :' + str(files[:val_num]))
            for f in files[:val_num]:
                if not f.startswith('.'):
                    shutil.move(os.path.join(train_data_dir, fod, f),
                                os.path.join(validate_data_dir, fod, f))


if __name__ == "__main__":
    # save_model_to_pb('ep012-loss100.753-val_loss90.289.h5', '.', 'yolov3')
    # split_data('train_data', 'validate_data', 0.2)

    convert_h5_to_pb('C:/Users/AlphaCat/Desktop/models/gmle_cnn_keras_classifier_resnet50/cnn_model/output',
                     'model_50_0.08.h5')
