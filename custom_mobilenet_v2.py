import tf_keras as keras

def make_divisible(v, divisor, min_value=None):
    # If no minimum value is specified, use the divisor as the minimum value.
    if min_value is None:
        min_value = divisor
    # Round the number of filters to be divisible by the divisor.
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that rounding down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def inverted_redsidual_block(input, filters, alpha, expansion_factor, strides=(1,1), id = True):
    
    pointwise_filters = int(filters * alpha)
    pointwise_filters = make_divisible(pointwise_filters, 8)


    shape = input.shape
    input_channels = shape[-1]
    
    x = input

    if id:
        x = keras.layers.Conv2D(expansion_factor*input_channels, kernel_size=(1,1), padding='same',use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU(6.0)(x)

    if strides == (2,2):
        # If the stride is 2, pad the input tensor to maintain the same output size.
        x = keras.layers.ZeroPadding2D(padding=keras.src.applications.imagenet_utils.correct_pad(x, 3))(x)
 
    x = keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding='same', strides=strides, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.0)(x)

    x = keras.layers.Conv2D(pointwise_filters, kernel_size=(1,1), padding='same',use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)


    if strides == (1,1) and input_channels == pointwise_filters:
        x = keras.layers.Add()([input,x])

    return x


def MobileNet_v2(input_shape, alpha, num_classes, dropout):

    input = keras.Input(shape=input_shape)

    first_block_filters =  make_divisible(32 * alpha, 8)
    #first_block_filters = 16


    x = keras.layers.Conv2D(first_block_filters, kernel_size= (3,3), strides=(2,2), padding='same', use_bias=False)(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.0)(x)

    x = inverted_redsidual_block(x, 16,     alpha, expansion_factor=1, id = False)

    x = inverted_redsidual_block(x, 24,     alpha, expansion_factor=6, strides=(2,2))
    x = inverted_redsidual_block(x, 24,     alpha, expansion_factor=6)

    x = inverted_redsidual_block(x, 32,     alpha, expansion_factor=6, strides=(2,2))
    x = inverted_redsidual_block(x, 32,     alpha, expansion_factor=6)
    x = inverted_redsidual_block(x, 32,     alpha, expansion_factor=6)

    x = inverted_redsidual_block(x, 64,     alpha, expansion_factor=6, strides=(2,2))
    x = inverted_redsidual_block(x, 64,     alpha, expansion_factor=6)
    x = inverted_redsidual_block(x, 64,     alpha, expansion_factor=6)
    x = inverted_redsidual_block(x, 64,     alpha, expansion_factor=6)

    x = inverted_redsidual_block(x, 96,     alpha, expansion_factor=6)
    x = inverted_redsidual_block(x, 96,     alpha, expansion_factor=6)
    x = inverted_redsidual_block(x, 96,     alpha, expansion_factor=6)

    x = inverted_redsidual_block(x, 160,    alpha, expansion_factor=6, strides=(2,2))
    x = inverted_redsidual_block(x, 160,    alpha, expansion_factor=6)
    x = inverted_redsidual_block(x, 160,    alpha, expansion_factor=6)

    x = inverted_redsidual_block(x, 320,    alpha, expansion_factor=6)

    x = keras.layers.Conv2D(1280, kernel_size=(1,1), padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.0)(x)

    
    x = keras.layers.GlobalAveragePooling2D()(x)

    if dropout:
        x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.Dense(32)(x)   
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dense(16)(x)
    x = keras.layers.ReLU()(x)

    if num_classes > 2:
            output = keras.layers.Dense(num_classes, activation='softmax')(x)
    else:
            output = keras.layers.Dense(1, activation='sigmoid')(x)

    
    model = keras.Model(input,output)

    return model
