# -*- coding: utf-8 -*-
import os
from PIL import Image
import numpy as np

# print(os.listdir('D:/Datasets/GuangDong/train_try'))

# print(os.listdir('))

im = Image.open('D:/Datasets/GuangDong/mask/outputs/attachments/脏点20180831094339对照样本_1.png')
# im.show()

img = np.asarray(im, dtype=np.uint8)
print(img.shape)

print(img[:, :, 3])

size = 512

x_start = np.random.randint(0, img.shape[0] - size - 1)
y_start = np.random.randint(0, img.shape[1] - size - 1)

x = img[x_start:x_start + size, y_start:y_start + size]

sum = np.sum(x)
print(sum)

im = Image.fromarray(x)
im.show()