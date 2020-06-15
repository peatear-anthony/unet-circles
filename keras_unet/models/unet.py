import logging
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, UpSampling2D, Cropping2D, Dropout, concatenate
from keras.utils import multi_gpu_model

def unet(input_shape=(1024,1024,3), num_of_classes = 3, dropout=0,
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
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(dropout)(conv4) if dropout else conv4
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Dropout(dropout)(conv5) if dropout else conv5
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Dropout(dropout)(conv6) if dropout else conv6
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Dropout(dropout)(conv7) if dropout else conv7
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Dropout(dropout)(conv8) if dropout else conv8
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Dropout(dropout)(conv9) if dropout else conv9
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(num_of_classes, (1, 1), activation='sigmoid')(conv9)
    
    if v2:
        model = Model(inputs=inputs, outputs=[conv10], name='Unet')
        
    else:
        model = Model(inputs=[inputs], outputs=[conv10], name='Unet')

    logger.info('Loading U-Net with shape: %s', str(input_shape))
    
    if multi_gpu and dual == False:
        model = multi_gpu_model(model, gpus=2)
        logger.info("Multi_gpu model activated!")
        return model
    
    elif dual:
        return inputs, conv9
    
    else:
        return model


if __name__ == '__main__':
    model = unet(multi_gpu=False)
    model.summary()