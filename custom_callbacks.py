from tensorflow.keras.callbacks import *
import tensorflow as tf
import keras.backend as K
from tensorflow.python.summary import summary as tf_summary


class CustomTensorBoard(TensorBoard):

    def __init__(self, write_input=False, max_result_display=3, **kwargs):
        self.write_input = write_input
        self.max_result_display = max_result_display
        super(CustomTensorBoard, self).__init__(**kwargs)

    def set_model(self, model):
        """
        Overwrite set_model method
        """
        super().set_model(model)

        if self.write_input:
            input_imgs = self.model.input

            assert len(K.int_shape(input_imgs)) == 4, 'Should be the 4-D images tensor [batch_size,height,width,channels]'
            tf.summary.image('Input_Image', input_imgs, max_outputs=self.max_result_display)

        self.merged = tf_summary.merge_all()
