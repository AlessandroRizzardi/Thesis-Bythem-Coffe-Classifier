import os;os.environ["TF_USE_LEGACY_KERAS"]="1"
import tf_keras as keras

def relu6(x):
    return keras.layers.ReLU(max_value=6)(x)

def hard_swish(x):
    return x * relu6(x + 3) / 6


def se_block(input, filters, se_ratio=0.25):
    se_shape = (1, 1, filters)
    se = keras.layers.GlobalAveragePooling2D()(input)
    se = keras.layers.Reshape(se_shape)(se)
    se = keras.layers.Dense(filters * se_ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = keras.layers.Dense(filters, activation='hard_sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    return keras.layers.Multiply()([input, se])


def mobilenet_v3_block(inputs, filters, kernel_size, strides, expansion_factor, alpha, se=False, activation='relu'):
    input_filters = inputs.shape[-1]
    expanded_filters = input_filters * expansion_factor
    
    x = keras.layers.Conv2D(int(expanded_filters * alpha), kernel_size=1, padding='same', use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = relu6(x) if activation == 'relu' else hard_swish(x)

    x = keras.layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = relu6(x) if activation == 'relu' else hard_swish(x)

    if se:
        x = se_block(x, int(expanded_filters * alpha))

    x = keras.layers.Conv2D(int(filters * alpha), kernel_size=1, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    
    if strides == 1 and input_filters == int(filters * alpha):
        x = keras.layers.Add()([inputs, x])

    return x


def MobileNetV3_Small(input_shape, number_classes, alpha=1, head=True, minimization=False):
    inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(int(16 * alpha), kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = hard_swish(x)

    x = mobilenet_v3_block(x, filters=16, kernel_size=3, strides=2, expansion_factor=1, alpha=alpha, se=True, activation='relu')
    x = mobilenet_v3_block(x, filters=24, kernel_size=3, strides=2, expansion_factor=72/16, alpha=alpha, se=False, activation='relu')
    x = mobilenet_v3_block(x, filters=24, kernel_size=3, strides=1, expansion_factor=88/24, alpha=alpha, se=False, activation='relu')
    x = mobilenet_v3_block(x, filters=40, kernel_size=5, strides=2, expansion_factor=4, alpha=alpha, se=True, activation='hard_swish')
    x = mobilenet_v3_block(x, filters=40, kernel_size=5, strides=1, expansion_factor=6, alpha=alpha, se=True, activation='hard_swish')
    x = mobilenet_v3_block(x, filters=48, kernel_size=5, strides=1, expansion_factor=3, alpha=alpha, se=True, activation='hard_swish')
    x = mobilenet_v3_block(x, filters=48, kernel_size=5, strides=1, expansion_factor=3, alpha=alpha, se=True, activation='hard_swish')
    x = mobilenet_v3_block(x, filters=96, kernel_size=5, strides=2, expansion_factor=6, alpha=alpha, se=True, activation='hard_swish')

    if minimization==False:
        x = mobilenet_v3_block(x, filters=96, kernel_size=5, strides=1, expansion_factor=6, alpha=alpha, se=True, activation='hard_swish')
        x = mobilenet_v3_block(x, filters=96, kernel_size=5, strides=1, expansion_factor=6, alpha=alpha, se=True, activation='hard_swish')

    x = keras.layers.Conv2D(int(576 * alpha), kernel_size=1, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = hard_swish(x)

    if head==True:
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Reshape((1, 1, int(576 * alpha)))(x)
        x = keras.layers.Conv2D(int(1024 * alpha), kernel_size=1, use_bias=False)(x)
        x = hard_swish(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Conv2D(number_classes, kernel_size=1)(x)
        outputs = keras.layers.Flatten()(x)
        model =keras.Model(inputs, outputs)
    else:
        model =keras.Model(inputs, x)
    return model




