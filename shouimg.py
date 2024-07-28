import tensorflow as tf
from keras.utils import image_dataset_from_directory, array_to_img
import matplotlib.pyplot as plt


def load_image_dataset(directory, image_size, batch_size, label_mode=None, color_mode='rgb', shuffle=None, seed=None):
    return image_dataset_from_directory(
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
    # Zip the image and mask datasets together
    dataset = tf.data.Dataset.zip((image_dataset, mask_dataset))

    # Combine the datasets to return images and masks as pairs
    def process(image, mask):
        return image, mask

    dataset = dataset.map(process, num_parallel_calls=tf.data.AUTOTUNE)
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


def display_image_mask_pairs(dataset, num_pairs=5):
    """
    显示数据集中图像和掩码对的函数。

    参数:
    dataset (tf.data.Dataset): 包含图像和掩码对的数据集。
    num_pairs (int): 要显示的图像和掩码对的数量。
    """
    plt.figure(figsize=(10, 2 * num_pairs))

    for i, (image, mask) in enumerate(dataset.take(num_pairs)):
        image = tf.squeeze(image, axis=0)  # 从批次中提取单个图像
        mask = tf.squeeze(mask, axis=0)  # 从批次中提取单个掩码

        plt.subplot(num_pairs, 2, 2 * i + 1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(image))
        plt.title("Image")
        plt.axis("off")

        plt.subplot(num_pairs, 2, 2 * i + 2)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(mask), cmap='gray')
        plt.title("Mask")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

display_image_mask_pairs(dataset)
