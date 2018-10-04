import os

from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K

from keras.preprocessing import image
import numpy as np


def focal_loss_category(gamma=2.):
    def focal_loss(y_true, y_pred):
        return - tf.reduce_sum(y_true * ((1 - y_pred) ** gamma) * tf.log(y_pred), axis=1)

    return focal_loss


model = load_model(
    'D:/Datasets/GuangDong/output-inc-res-90/model_80_0.04.h5',
    custom_objects={'focal_loss_fixed': focal_loss_category()})

model.summary()
print(model.layers)
img_width = 299
img_height = 299

# img_path = 'C:/Users/AlphaCat/Desktop/models/gmle_cnn_keras_classifier_resnet50/tf_keras/555.jpg'
directory = 'D:/Datasets/GuangDong/test'

for img_name in os.listdir(directory):
    img_path = os.path.join(directory, img_name)
    img = image.load_img(img_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer("conv_7b_ac")
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(len(pooled_grads_value)):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    new_img_name = img_name.split('.')[0] + '_cam_' + str(class_idx) + '.jpg'
    cv2.imwrite(os.path.join(directory, new_img_name), superimposed_img)

print('Done')
