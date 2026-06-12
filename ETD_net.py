from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Add, LeakyReLU, Concatenate, DepthwiseConv2D, \
    BatchNormalization, UpSampling2D, Dropout, Multiply
from keras.optimizers import Adam
from keras import backend as K
from load import vgg_block, det_deconv, CAM, PAM, srm_init
import densenet
import cbam
import tensorflow as tf
from keras.layers import Lambda


def weighted_binary_crossentropy(pos_weight):

    def loss(y_true, y_pred):

        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        pos_loss = y_true * K.log(y_pred)

        neg_loss = (1 - y_true) * K.log(1 - y_pred)

        weighted_pos_loss = pos_weight * pos_loss

        loss_val = - (weighted_pos_loss + neg_loss)

        return K.mean(loss_val)

    return loss


def model(gpus=1):
    def conv(x, filters, kernel=3, strides=1, dilation=1):
        x = Conv2D(filters=filters, kernel_size=kernel, strides=strides, dilation_rate=dilation, padding='same',
                   activation='relu')(x)
        return x

    rgb = Input(shape=(None, None, 3))

    coarse_in = DepthwiseConv2D(depth_multiplier=3, kernel_size=(5, 5), padding='same', depthwise_initializer=srm_init)(
        rgb)
    coarse_in.trainable = False
    x = coarse_in

    growth_rate = 4
    x = densenet.DenseBlock(x, 4, growth_rate)
    x = densenet.TransitionLayer(x)
    x1 = x

    x = cbam.CBAM_block(x)

    def frequency_attention(x):
        input_shape = K.int_shape(x)
        channels = input_shape[-1]

        freq_attention = Lambda(lambda t: K.mean(t, axis=2, keepdims=True))(x)
        freq_attention = Conv2D(channels // 8, (1, 1), activation='relu', padding='same')(freq_attention)

        freq_attention = Conv2D(channels, (1, 1), activation='sigmoid', padding='same', name='freq_attention_weights')(
            freq_attention)

        output = Multiply()([x, freq_attention])
        return output

    x = frequency_attention(x)


    x = densenet.DenseBlock(x, 2, growth_rate)
    x = densenet.TransitionLayer(x)
    x2 = x

    x = vgg_block(x, filters=256, is_seven=True)

    dilated_2 = conv(x, 256, dilation=2)
    dilated_4 = conv(x, 256, dilation=4)
    dilated_8 = conv(x, 256, dilation=8)
    dilated_16 = conv(x, 256, dilation=16)

    x = Add()([dilated_2, dilated_4, dilated_8, dilated_16])

    pam = PAM()(x)
    pam = Conv2D(256, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(pam)
    cam = CAM()(x)
    cam = Conv2D(256, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(cam)
    feature_sum = Add()([pam, cam])
    feature_sum = Conv2D(256, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(feature_sum)
    x3 = feature_sum

    x3 = Concatenate(name='concat1')([x3, x2])

    x4 = det_deconv(x3, 64)
    x5 = vgg_block(x4, filters=64)
    x5 = Concatenate(name='concat2')([x1, x5])
    x6 = det_deconv(x5, 32)

    out = Conv2D(1, 7, padding='same', activation='sigmoid', name='out1')(x6)

    model = Model(inputs=[rgb], outputs=out)

    for layer in model.layers:
        if 'depthwise_conv2d' in layer.name:
            layer.trainable = False

    coarse_in.trainable = False

    optimizer = Adam(lr=0.0001, clipvalue=0.5)

    pos_weight_value = 1.0 / 3.0

    model.compile(
        optimizer=optimizer,
        loss=weighted_binary_crossentropy(pos_weight_value),
        metrics=['accuracy']
    )

    return model
