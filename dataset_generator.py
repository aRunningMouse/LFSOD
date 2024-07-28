import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from glob import glob

def load_image(path, size, mode):
    """Load an image from a file, resize, and normalize it."""
    image = keras.utils.load_img(path, target_size=size, color_mode=mode)
    image = keras.utils.img_to_array(image)
    image = (image / 255.0).astype(np.float32)
    return image

def preprocess(x_batch, y_batch, input_size, output_size, out_classes):
    """Preprocess function to load and resize images and masks."""
    def f(image_path, mask_path):
        image_path, mask_path = image_path.decode(), mask_path.decode()
        image = load_image(image_path, input_size, mode="rgb")  # Load and resize image
        mask = load_image(mask_path, output_size, mode="grayscale")  # Load and resize mask
        return image, mask

    images, masks = tf.numpy_function(f, [x_batch, y_batch], [tf.float32, tf.float32])
    images.set_shape([input_size[0], input_size[1], 3])
    masks.set_shape([output_size[0], output_size[1], out_classes])
    return images, masks

def dataset_generator(image_dir, mask_dir, input_size, output_size, out_classes, batch_size, shuffle=True):
    """Create a TensorFlow dataset generator for images and masks."""
    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    
    if shuffle:
        dataset = dataset.cache().shuffle(buffer_size=1000)
    
    dataset = dataset.map(
        lambda x, y: preprocess(x, y, input_size, output_size, out_classes),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Example usage
image_dir = r"D:\dlut-dataset\test_data\test_array"
mask_dir = r"D:\dlut-dataset\test_data\test_masks"
input_size = (256*9, 256*9)  # Input image size
output_size = (288, 288)  # Output mask size
out_classes = 1  # Assuming binary masks, use 1 class. Adjust if needed
batch_size = 1

# Create the dataset generator
dataset = dataset_generator(image_dir, mask_dir, input_size, output_size, out_classes, batch_size)

# Iterate through the dataset (for demonstration purposes)



