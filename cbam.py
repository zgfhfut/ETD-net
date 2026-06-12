import keras
from keras.models import *
from keras.layers import *
from keras import layers
import keras.backend as K

IMAGE_ORDERING = 'channels_last'
def CBAM_block(cbam_feature,ratio=8):
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def channel_attention(input_feature,ratio=8):

    channel = input_feature._keras_shape[-1]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPool2D()(input_feature)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam)

    return multiply([input_feature,cbam_feature])

def spatial_attention(input_feature):

    avg_pool = Lambda(lambda x:K.mean(x,axis=3,keepdims=True))(input_feature)
    max_pool = Lambda(lambda x:K.max(x,axis=3,keepdims=True))(input_feature)

    concat = Concatenate(axis=3)([avg_pool,max_pool])
    cbam_feature = Conv2D(1,(7,7),strides=1,padding='same',activation='sigmoid')(concat)

    return multiply([input_feature,cbam_feature])
