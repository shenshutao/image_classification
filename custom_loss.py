import tensorflow as tf


def focal_loss_category(gamma=2.):
    def focal_loss(y_true, y_pred):
        return - tf.reduce_sum(y_true * ((1. - y_pred) ** gamma) * tf.log(y_pred), axis=1)

    return focal_loss


def focal_loss_category2(gamma=2.):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return - tf.reduce_sum(tf.pow(1. - pt_1, gamma) * tf.log(pt_1), axis=1) \
               - tf.reduce_sum(tf.pow(pt_0, gamma) * tf.log(1. - pt_0), axis=1)

    return focal_loss_fixed


if __name__ == '__main__':
    logits = tf.random_uniform(shape=[5, 4], minval=-1, maxval=1, dtype=tf.float32)
    labels = tf.Variable([1, 0, 2, 3, 1])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y_pred2 = tf.nn.softmax(logits, dim=-1)  # [batch_size,num_classes]
        y_true = tf.one_hot(labels, depth=y_pred2.shape[1])

        func1 = focal_loss_category()
        loss1 = func1(y_true, y_pred2)

        func2 = focal_loss_category2()
        loss2 = func2(y_true, y_pred2)

        print(loss1)
        print(loss2)
