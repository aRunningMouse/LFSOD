import tensorflow as tf
from tensorflow import keras 
from keras import layers
from keras.layers import Conv2D, BatchNormalization , Activation, DepthwiseConv2D, Concatenate


print(tf.__version__)

"""

输入图片大小：256*9,256*9,3 -> 2304,2304,3 
假设初始 filter=32， batch_size=1
最后一个卷积层的大小 (144,144,256), batch_size=1


"""


def large_kernel(filters,input_image,dilation_rate=1):
    x = DepthwiseConv2D(kernel_size=(9, 9), padding='same', activation='relu',dilation_rate=dilation_rate)(input_image)
    x = Conv2D(filters=filters,activation='relu',padding='same',dilation_rate=1,kernel_size=(5,5))(x)
    x = BatchNormalization()(x)
    return x


from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, Activation


def conv_block(input_image, dilation_rate, filters):
    # Depthwise convolution with dilation
    y1 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', dilation_rate=dilation_rate)(input_image)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)

    # Pointwise convolution (1x1) to match the number of filters
    y1 = Conv2D(filters=filters, kernel_size=(1, 1), padding='same')(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)

    # Depthwise convolution without dilation
    y1 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', dilation_rate=1)(y1)
    y1 = BatchNormalization()(y1)

    # Pointwise convolution (1x1) to match the number of filters
    y2 = Conv2D(filters=filters, kernel_size=(1, 1), padding='same')(y1)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)

    return y2


def my_large_kernel(input_image,dilation_rate,filters,):

    filters = int(filters)

########################### 交互层1 ###################################################


    x1 = large_kernel(filters,input_image=input_image,dilation_rate=int(dilation_rate))
    x1 = keras.layers.MaxPooling2D()(x1)

    y1 = conv_block(input_image,dilation_rate=1,filters=filters)
    y1_1 = keras.layers.MaxPooling2D()(y1)

    inter1_1 = Concatenate()([x1,y1_1]) #256 -> 128
    inter1_1 = Conv2D(filters=filters,activation='relu',padding='same',kernel_size=(1,1))(inter1_1)

########################### 交互层2 ###################################################


    x2 = large_kernel(2*filters,x1,dilation_rate=int(dilation_rate/2))
    x2 = layers.MaxPooling2D()(x2)

    y2 = conv_block(inter1_1,dilation_rate=1,filters=2*filters) #72
    y2_1 = layers.MaxPooling2D()(y2)

    inter2_1 = Concatenate()([x2,y2_1]) #128 -> 64
    inter2_1 = layers.Conv2D(2*filters,padding='same',activation='relu',kernel_size=(1,1))(inter2_1)  # 256 -> 64

########################### 交互层3 ###################################################


    x3 = large_kernel(4*filters,input_image=x2,dilation_rate=int(dilation_rate/4))
    x3 = keras.layers.MaxPooling2D()(x3)


    y3 = conv_block(input_image=inter2_1,dilation_rate=1,filters= 4 * filters) #36
    y3_1 = keras.layers.MaxPooling2D()(y3)

    inter3_1 = Concatenate()([x3,y3_1]) # 64 -> 32
    inter3_1 = Conv2D(filters= 4 * filters,activation='relu',padding='same',kernel_size=(1,1))(inter3_1)

    ########################### 交互层4 ###################################################

    x4 = large_kernel(8*filters,x3,dilation_rate=int(dilation_rate/8))                                              #18
    x4 = layers.MaxPooling2D()(x4)

    y4 = conv_block(inter3_1,dilation_rate=1,filters=8 * filters)
    y4_1 = layers.MaxPooling2D()(y4)

    inter4_1 = Concatenate()([x4,y4_1]) # 32 -> 16
    inter4_1 = layers.Conv2D( 8*filters,padding='same',activation='relu',kernel_size=(1,1))(inter4_1)  # 64 -> 16   #9
    return inter4_1




def my_model():
    input_my = keras.Input(shape=(256*9,256*9,3))

    mid = my_large_kernel(input_image=input_my,dilation_rate=256,filters=16)
    out1 = Conv2D(kernel_size=(1,1),filters=16*4, activation='relu',padding='same')(mid)
    out1 = Conv2D(kernel_size=(1, 1), filters=16 * 2, activation='relu', padding='same')(out1)
    out1 = Conv2D(kernel_size=(1, 1), filters=16 * 1, activation='relu', padding='same')(out1)
    out1 = Conv2D(kernel_size=(1, 1), filters=8 , activation='relu', padding='same')(out1)
    output_my = keras.layers.UpSampling2D((2,2))(out1)
    output_my = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same')(output_my)
    model = keras.Model(input_my,output_my)

    return model 


model = my_model()
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy')



import os
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from glob import glob

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
input_size = (256*9, 256*9)  # Input image size
output_size = (288, 288)  # Output mask size
batch_size = 1

# Load image and mask datasets
image_dataset = load_image_dataset(image_dir, input_size, batch_size, color_mode='rgb')
mask_dataset = load_image_dataset(mask_dir, output_size, batch_size, color_mode='grayscale')

# Combine the datasets
dataset = combine_image_mask_datasets(image_dataset, mask_dataset)



# model.summary()
model.summary()