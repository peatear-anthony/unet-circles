import logging
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, UpSampling2D, Cropping2D, Dropout, concatenate
from keras.utils import multi_gpu_model

def unet_mini(input_shape=(1024,1024,1), num_of_classes = 3, dropout=0,
            multi_gpu=False, dual=False, input_layer=None, inputs=None, v2=False):

    logger = logging.getLogger(__name__)


    # For input_layer from unet_de_v2
    if v2:
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)

    else:
        inputs = Input(input_shape)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)

    conv1 = Dropout(dropout)(conv1) if dropout else conv1
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(dropout)(conv2) if dropout else conv2
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(dropout)(conv3) if dropout else conv3
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up4 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv3), conv2], axis=3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up4)
    conv4 = Dropout(dropout)(conv6) if dropout else conv4
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    up5 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4), conv1], axis=3)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up5)
    conv5 = Dropout(dropout)(conv7) if dropout else conv5
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    conv6 = Conv2D(num_of_classes, (1, 1), activation='sigmoid')(conv5)
    
    if v2:
        model = Model(inputs=inputs, outputs=[conv6], name='unet_mini')
        
    else:
        model = Model(inputs=[inputs], outputs=[conv6], name='unet_mini')

    logger.info('Loading U-Net with shape: %s', str(input_shape))
    
    if multi_gpu and dual == False:
        model = multi_gpu_model(model, gpus=2)
        logger.info("Multi_gpu model activated!")
        return model
    
    elif dual:
        return inputs, conv5
    
    else:
        return model


if __name__ == '__main__':
    model = unet_mini(multi_gpu=False)
    model.summary()