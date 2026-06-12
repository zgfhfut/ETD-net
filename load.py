import cv2
import numpy as np
import keras
import numpy as np
import tensorflow as tf
import glob
from keras.layers import Conv2D, UpSampling2D, InputLayer, Lambda, Input, Reshape, Activation, Dense, Dropout, Flatten, ELU
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers import MaxPooling2D,AveragePooling2D,UpSampling2D,SeparableConv2D,LeakyReLU,Concatenate,DepthwiseConv2D,Add
from keras import backend as K
from keras.layers import Layer
import keras.layers as layers
base_path = ''


def load_rgb(line):
    size = (640, 480)

    rgb = cv2.imread(base_path + line.split(' ')[0])
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    rgb = np.array(rgb) / 255.
    # print rgb
    rgb = rgb[:, :, ::-1]

    return rgb


def load_mask(line):
    size = (640, 480)

    mask = cv2.imread(base_path + line.split(' ')[1])
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    mask = np.expand_dims(np.mean(mask, axis=-1), axis=-1)
    mask = np.array(mask) / 255.

    return mask


def srm_init(shape, dtype=None):

    hpf = np.zeros(shape, dtype=np.float32)

    hpf[:, :, 0, 0] = np.array(
        [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]]) / 4.0
    hpf[:, :, 0, 1] = np.array(
        [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]) / 12.
    hpf[:, :, 0, 2] = np.array(
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) / 2.0

    return hpf


class PAM(Layer):
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(PAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

        self._depthwise_conv = None
        self._conv_256 = None
        self._conv_b = None
        self._conv_c = None
        self._conv_d = None
        self._input_shape_cache = None

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1,),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def _get_or_create_layers(self, input_shape, filters):

        cache_key = (input_shape[1], input_shape[2], filters)

        if self._input_shape_cache != cache_key:

            self._depthwise_conv = DepthwiseConv2D(
                depth_multiplier=3, 
                kernel_size=(5, 5), 
                padding='same', 
                depthwise_initializer=srm_init,
                trainable=False
            )

            self._depthwise_conv.build(input_shape)

            depthwise_output_shape = list(input_shape)
            depthwise_output_shape[-1] = input_shape[-1] * 3
            self._conv_256 = Conv2D(256, 1, use_bias=False, kernel_initializer='he_normal')
            self._conv_256.build(tuple(depthwise_output_shape))

            conv_256_output_shape = list(depthwise_output_shape)
            conv_256_output_shape[-1] = 256
            self._conv_b = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')
            self._conv_b.build(tuple(conv_256_output_shape))
            self._conv_c = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')
            self._conv_c.build(tuple(conv_256_output_shape))
            self._conv_d = Conv2D(filters, 1, use_bias=False, kernel_initializer='he_normal')
            self._conv_d.build(tuple(conv_256_output_shape))
            
            self._input_shape_cache = cache_key

    def call(self, input):
        origin = input

        input_shape = origin.get_shape().as_list()
        _, h, w, filters = input_shape

        self._get_or_create_layers(input_shape, filters)

        x = self._depthwise_conv(input)
        x = self._conv_256(x)
        
        h = tf.shape(origin)[1]
        w = tf.shape(origin)[2]

        b = self._conv_b(x)
        c = self._conv_c(x)
        d = self._conv_d(x)

        vec_b = K.reshape(b, (-1, h * w, filters // 8))
        vec_cT = tf.transpose(K.reshape(c, (-1, h * w, filters // 8)), (0, 2, 1))
        bcT = K.batch_dot(vec_b, vec_cT)
        softmax_bcT = Activation('sigmoid')(bcT)

        vec_d = K.reshape(d, (-1, h * w, filters))
        bcTd = K.batch_dot(softmax_bcT, vec_d)
        bcTd = K.reshape(bcTd, (-1, h, w, filters))

        out = self.gamma * bcTd + origin
        return out


class CAM(Layer):
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(CAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

        self._depthwise_conv = None
        self._conv_256 = None
        self._input_shape_cache = None

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(1,),
            initializer=self.gamma_initializer,
            name='gamma',
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint,
        )
        super(CAM, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def _get_or_create_layers(self, input_shape):

        cache_key = (input_shape[1], input_shape[2], input_shape[3])

        if self._input_shape_cache != cache_key:

            self._depthwise_conv = DepthwiseConv2D(
                depth_multiplier=3,
                kernel_size=(5, 5),
                padding='same',
                depthwise_initializer=srm_init
            )

            self._depthwise_conv.build(input_shape)

            depthwise_output_shape = list(input_shape)
            depthwise_output_shape[-1] = input_shape[-1] * 3
            self._conv_256 = Conv2D(
                256,
                1,
                use_bias=False,
                kernel_initializer='he_normal'
            )
            self._conv_256.build(tuple(depthwise_output_shape))
            
            self._input_shape_cache = cache_key

    def call(self, input):
        origin = input

        input_shape = origin.get_shape().as_list()

        self._get_or_create_layers(input_shape)

        x = self._depthwise_conv(input)
        x = self._conv_256(x)

        x_shape = K.int_shape(x)
        _, h_static, w_static, filters = x_shape
        filters = filters

        h = tf.shape(origin)[1]
        w = tf.shape(origin)[2]

        vec_a = K.reshape(x, (-1, h * w, filters))

        vec_aT = tf.transpose(vec_a, (0, 2, 1))

        aTa = K.batch_dot(vec_aT, vec_a)

        softmax_aTa = tf.nn.sigmoid(aTa)

        aaTa = K.batch_dot(vec_a, softmax_aTa)

        aaTa = K.reshape(aaTa, (-1, h, w, filters))

        out = self.gamma * aaTa + origin
        return out


def vgg_block(x, filters, pooling=False, is_seven=False, last=False, name='out1'):

    x = layers.Conv2D(filters, (3, 3),
                      activation='relu',
                      padding='same')(x)
    x = layers.Conv2D(filters, (3, 3),
                      activation='relu',
                      padding='same')(x)
    x = layers.Conv2D(filters, (3, 3),
                      activation='relu',
                      padding='same')(x)
    if is_seven:
        x = layers.Conv2D(filters, (3, 3),
                          padding='same')(x)
        if last:
            x = Activation('sigmoid', name=name)(x)
        else:
            x = Activation('relu')(x)
    if pooling:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    return x


def det_deconv(x,filters):
    x = UpSampling2D()(x)
    x = Conv2D(filters,3,activation='relu',padding='same')(x)
    return x


def load_img(filelines):
    X = []
    Y = []
    O = []
    size = (480, 640)
    for line in filelines:

        try:
            rgb = cv2.imread(base_path + line.split(' ')[0])
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            rgb = rgb[:, :, ::-1]

            mask = cv2.imread(base_path + line.split(' ')[1])  # ,cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(np.mean(mask, axis=-1), axis=-1)
        except:
            continue
        rgb = list(rgb)
        mask = list(mask)
        X.append(np.array(rgb)/255.)
        Y.append((np.array(mask)/255.).astype(np.uint))

    return X, Y
