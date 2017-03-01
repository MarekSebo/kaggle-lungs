import numpy as np
import os
import tensorflow as tf
from build_model import BuildModel, accuracy
import logging
from loading import DataClass

model_name = 'prvy'
checkpoints = [1]
batch_size = 1
chunk_size = 1
info_freq = 20
augmentation = False
learning_rate = 0.001
image_height, image_width, channels = (512, 512, 3)

models_dir = 'models'
data_dir = 'data'
labels_dir = ''
print("root directory is:", os.getcwd())

# create models dir
if not os.path.exists(models_dir):
    os.mkdir(models_dir)


url = os.getcwd()
print(os.path.join(url, data_dir))
train_data = DataClass(os.path.join(url, data_dir, '/'), os.path.join(url, data_dir, '/'), batch_size, chunk_size,
                       image_height, image_width, augm=augmentation, data_use='train')




