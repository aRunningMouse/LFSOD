import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, DepthwiseConv2D, Concatenate

print(tf.__version__)


def large_kernel(filters, input_image, dilation_rate=1):
    x = tf.nn.relu(DepthwiseConv2D(kernel_size=(9, 9), padding='same', dilation_rate=dilation_rate)(input_image))
    x = tf.nn.relu(Conv2D(filters=filters, padding='same', dilation_rate=1, kernel_size=(5, 5))(x))
    x = BatchNormalization()(x)
    return x


def conv_block(input_image, dilation_rate, filters):
    y1 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', dilation_rate=dilation_rate)(input_image)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', dilation_rate=1)(y1)
    y1 = BatchNormalization()(y1)
    y2 = Activation('relu')(y1)
    return y2


def my_large_kernel(input_image, dilation_rate, filters):
    filters = int(filters)

    # Interaction Layer 1
    x1 = large_kernel(filters, input_image=input_image, dilation_rate=int(dilation_rate))
    x1 = tf.nn.max_pool2d(x1, ksize=2, strides=2, padding='SAME')

    y1 = conv_block(input_image, dilation_rate=1, filters=filters)
    y1_1 = tf.nn.max_pool2d(y1, ksize=2, strides=2, padding='SAME')

    inter1_1 = Concatenate()([x1, y1_1])
    inter1_1 = Conv2D(filters=filters, padding='same', kernel_size=(1, 1), activation='relu')(inter1_1)

    # Interaction Layer 2
    x2 = large_kernel(2 * filters, x1, dilation_rate=int(dilation_rate / 2))
    x2 = tf.nn.max_pool2d(x2, ksize=2, strides=2, padding='SAME')

    y2 = conv_block(inter1_1, dilation_rate=1, filters=2 * filters)
    y2_1 = tf.nn.max_pool2d(y2, ksize=2, strides=2, padding='SAME')

    inter2_1 = Concatenate()([x2, y2_1])
    inter2_1 = Conv2D(2 * filters, padding='same', kernel_size=(1, 1), activation='relu')(inter2_1)

    # Interaction Layer 3
    x3 = large_kernel(4 * filters, input_image=x2, dilation_rate=int(dilation_rate / 4))
    x3 = tf.nn.max_pool2d(x3, ksize=2, strides=2, padding='SAME')

    y3 = conv_block(inter2_1, dilation_rate=1, filters=4 * filters)
    y3_1 = tf.nn.max_pool2d(y3, ksize=2, strides=2, padding='SAME')

    inter3_1 = Concatenate()([x3, y3_1])
    inter3_1 = Conv2D(filters=4 * filters, padding='same', kernel_size=(1, 1), activation='relu')(inter3_1)

    # Interaction Layer 4
    x4 = large_kernel(8 * filters, x3, dilation_rate=int(dilation_rate / 8))
    x4 = tf.nn.max_pool2d(x4, ksize=2, strides=2, padding='SAME')

    y4 = conv_block(inter3_1, dilation_rate=1, filters=8 * filters)
    y4_1 = tf.nn.max_pool2d(y4, ksize=2, strides=2, padding='SAME')

    inter4_1 = Concatenate()([x4, y4_1])
    inter4_1 = Conv2D(filters=8 * filters, padding='same', kernel_size=(1, 1), activation='relu')(inter4_1)

    return inter4_1


def image_cal(input_x):
    x = input_x
    x_out = my_large_kernel(input_image=x, dilation_rate=256, filters=64)
    y1 = tf.image.resize(x_out, (x_out.shape[1] * 2, x_out.shape[2] * 2))
    out_img = Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')(y1)
    return out_img


def my_model():
    input_my = tf.keras.Input(shape=(256 * 9, 256 * 9, 3))
    mid = my_large_kernel(input_image=input_my, dilation_rate=256, filters=16)
    output_my = tf.image.resize(mid, (mid.shape[1] * 2, mid.shape[2] * 2))
    output_my = Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')(output_my)
    model = tf.keras.Model(input_my, output_my)
    return model


model = my_model()
model.compile(optimizer='rmsprop', loss='binary_crossentropy')

import os
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from glob import glob


def load_image_dataset(directory, image_size, batch_size, label_mode=None, color_mode='rgb', shuffle=None, seed=None):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels=None,  # No labels needed for image and mask datasets
        label_mode=label_mode,
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed
    )


def combine_image_mask_datasets(image_dataset, mask_dataset):
    dataset = tf.data.Dataset.zip((image_dataset, mask_dataset))
    dataset = dataset.map(lambda image, mask: (image, mask), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# Example usage
image_dir = r"D:\dlut-dataset\all_test\ta"
mask_dir = r"D:\dlut-dataset\all_test\tm"
input_size = (256 * 9, 256 * 9)  # Input image size
output_size = (288, 288)  # Output mask size
batch_size = 1

# Load image and mask datasets
image_dataset = load_image_dataset(image_dir, input_size, batch_size, color_mode='rgb')
mask_dataset = load_image_dataset(mask_dir, output_size, batch_size, color_mode='grayscale')

# Combine the datasets
dataset = combine_image_mask_datasets(image_dataset, mask_dataset)

# Print model summary
model.summary()
