import matplotlib.pyplot as plt
import tensorflow as tf

## 编解码
# 读取图片
image_raw_data = tf.gfile.FastGFile("cat.jpg", 'rb').read()

with tf.Session() as sess:
        # 对于PNG图片，调用decode_png
    img_data = tf.image.decode_jpeg(image_raw_data)

    # 输出解码之后的三维矩阵。
    print(img_data.eval())

    plt.imshow(img_data.eval())
    plt.show()

    # 重新编码
    # 将数据转化为实数，以方便下面的示例程序对图像进行处理
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.uint8)

    encodeed_image = tf.image.encode_png(img_data)
    with tf.gfile.GFile("cat.png", "wb") as f:
        f.write(encodeed_image.eval())


## 大小调整
import matplotlib.pyplot as plt
import tensorflow as tf

# 读取图片
image_raw_data = tf.gfile.FastGFile("cat.jpg", 'rb').read()

with tf.Session() as sess:
    # 对于PNG图片，调用decode_png
    img_data = tf.image.decode_jpeg(image_raw_data)
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

    # 直接缩放成任意大小
    resized = tf.image.resize_images(img_data, [300, 300], method=0)
    plt.imshow(resized.eval())
    plt.show()

    # 使用裁剪和填充
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 300, 300)
    plt.imshow(croped.eval())
    plt.show()

    padded = tf.image.resize_image_with_crop_or_pad(img_data, 3000, 3000)
    plt.imshow(padded.eval())
    plt.show()

    # 通过比例裁剪
    central_cropped = tf.image.central_crop(img_data, 0.5)
    plt.imshow(central_cropped.eval())
    plt.show()


## 图像翻转
import matplotlib.pyplot as plt
import tensorflow as tf

# 读取图片
image_raw_data = tf.gfile.FastGFile("cat.jpg", 'rb').read()

with tf.Session() as sess:
    # 对于PNG图片，调用decode_png
    img_data = tf.image.decode_jpeg(image_raw_data)

    # 上下翻转
    flipped = tf.image.flip_up_down(img_data)
    plt.imshow(flipped.eval())
    plt.show()

    # 左右翻转
    flipped = tf.image.flip_left_right(img_data)
    plt.imshow(flipped.eval())
    plt.show()

    # 左右翻转
    transposed = tf.image.transpose_image(img_data)
    plt.imshow(transposed.eval())
    plt.show()

    # 概率上下翻转
    flipped = tf.image.random_flip_up_down(img_data)
    plt.imshow(flipped.eval())
    plt.show()

    # 概率左右翻转
    flipped = tf.image.random_flip_left_right(img_data)
    plt.imshow(flipped.eval())
    plt.show()

## import matplotlib.pyplot as plt
import tensorflow as tf

# 读取图片
image_raw_data = tf.gfile.FastGFile("cat.jpg", 'rb').read()

with tf.Session() as sess:
    # 对于PNG图片，调用decode_png
    img_data = tf.image.decode_jpeg(image_raw_data)

    # 亮度调整 -0.5
    adjusted = tf.image.adjust_brightness(img_data, -0.5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 亮度调整 +0.5
    adjusted = tf.image.adjust_brightness(img_data, 0.5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 亮度随机调整
    adjusted = tf.image.random_brightness(img_data, 0.5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 对比度调整 -5
    adjusted = tf.image.adjust_contrast(img_data, -5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 对比度调整 +5
    adjusted = tf.image.adjust_contrast(img_data, 5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 对比度随机调整
    adjusted = tf.image.random_contrast(img_data, 1, 5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 色相调整 0.1
    adjusted = tf.image.adjust_hue(img_data, 0.1)
    plt.imshow(adjusted.eval())
    plt.show()

    # 色相调整 0.5
    adjusted = tf.image.adjust_hue(img_data, 0.5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 色相随机调整
    adjusted = tf.image.random_hue(img_data, 0.5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 饱和度调整 -5
    adjusted = tf.image.adjust_saturation(img_data, -5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 饱和度调整 +5
    adjusted = tf.image.adjust_saturation(img_data, 5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 饱和度随机调整
    adjusted = tf.image.random_saturation(img_data, 1, 5)
    plt.imshow(adjusted.eval())
    plt.show()

    # 均值为0，方差为1
    adjusted = tf.image.per_image_standardization(img_data)
    plt.imshow(adjusted.eval())
    plt.show()
