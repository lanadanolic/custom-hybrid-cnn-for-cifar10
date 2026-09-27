# CNN architectures used in the project

from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10


# data augmentation that is applied during training
def make_data_augmentation():
    return keras.Sequential([
        layers.RandomFlip(
            "horizontal",
            seed=SEED
        ),
        layers.RandomTranslation(
            0.1,
            0.1,
            fill_mode="reflect",
            seed=SEED + 1
        )
    ])


# Standard CNN baseline model (all convolutional blocks use standard Conv2D layers)
def build_standard_cnn():

    inputs = keras.Input(shape=INPUT_SHAPE)
    x = make_data_augmentation()(inputs)

    # block 1 - extracts basic image features using 32 filters
    x = layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    # normalizes the output and applies the ReLU activation function
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # block 2 - increases the number of feature maps to 48
    x = layers.Conv2D(
        filters=48,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # reduces the spatial dimensions of the feature maps
    x = layers.MaxPooling2D(pool_size=2)(x)

    # block 3 - extracts more complex features using 96 filters
    x = layers.Conv2D(
        filters=96,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # block 4 - final convolutional block with 192 filters
    x = layers.Conv2D(
        filters=192,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # converts the feature maps into one feature vector
    x = layers.GlobalAveragePooling2D()(x)

    # helps reduce overfitting
    x = layers.Dropout(0.3)(x)

    # final classification layer for the 10 CIFAR-10 classes
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="StandardCNN"
    )


# creates the Depthwise CNN baseline model (it has the same general structure as the Standard CNN, but uses depthwise separable convolutions)
def build_depthwise_cnn():

    inputs = keras.Input(shape=INPUT_SHAPE)
    x = make_data_augmentation()(inputs)

    # block 1 - depthwise separable convolution with 32 filters
    x = layers.SeparableConv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # block 2 - increases the number of feature maps to 48
    x = layers.SeparableConv2D(
        filters=48,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # block 3 - uses 96 filters
    x = layers.SeparableConv2D(
        filters=96,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # block 4 - final convolutional block with 192 filters
    x = layers.SeparableConv2D(
        filters=192,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # converts the feature maps into one feature vector
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # final classification layer
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="DepthwiseCNN"
    )


# creates the Dilated CNN baseline model (it uses dilated convolutions with dilation rate 2 to increase the receptive field)
def build_dilated_cnn():

    inputs = keras.Input(shape=INPUT_SHAPE)
    x = make_data_augmentation()(inputs)

    # block 1 - dilated convolution with 32 filters
    x = layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        dilation_rate=2,
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # block 2 - dilated convolution with 48 filters
    x = layers.Conv2D(
        filters=48,
        kernel_size=3,
        padding="same",
        dilation_rate=2,
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # block 3 - dilated convolution with 96 filters
    x = layers.Conv2D(
        filters=96,
        kernel_size=3,
        padding="same",
        dilation_rate=2,
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # block 4 - final dilated convolution with 192 filters
    x = layers.Conv2D(
        filters=192,
        kernel_size=3,
        padding="same",
        dilation_rate=2,
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # final classification layer
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="DilatedCNN"
    )


# creates one block of the proposed Hybrid CNN (each block processes the same input through three parallel branches: standard, depthwise separable, and dilated convolution)
def hybrid_block(
    x,
    filters_per_branch,
    output_filters,
    dilation_rate=2
):

    # standard convolution branch
    standard = layers.Conv2D(
        filters=filters_per_branch,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    standard = layers.BatchNormalization()(standard)
    standard = layers.ReLU()(standard)

    # depthwise separable convolution branch
    depthwise = layers.SeparableConv2D(
        filters=filters_per_branch,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(x)

    depthwise = layers.BatchNormalization()(depthwise)
    depthwise = layers.ReLU()(depthwise)

    # dilated convolution branch
    dilated = layers.Conv2D(
        filters=filters_per_branch,
        kernel_size=3,
        padding="same",
        dilation_rate=dilation_rate,
        use_bias=False
    )(x)

    dilated = layers.BatchNormalization()(dilated)
    dilated = layers.ReLU()(dilated)

    # combines the feature maps produced by all three branches
    x = layers.Concatenate()([
        standard,
        depthwise,
        dilated
    ])

    # a 1x1 convolution learns how to combine the features obtained from the three different convolution types
    x = layers.Conv2D(
        filters=output_filters,
        kernel_size=1,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x


# creates the proposed custom Hybrid CNN (the model is built from four hybrid blocks that combine standard, depthwise separable, and dilated convolutions)
def build_hybrid_cnn(dilation_rate=2):
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = make_data_augmentation()(inputs)

    # block 1
    # 11 filters are used in each branch and the combined feature maps are transformed into 32 output feature maps
    x = hybrid_block(
        x,
        filters_per_branch=11,
        output_filters=32,
        dilation_rate=1
    )

    # block 2
    # 16 filters per branch produce 48 feature maps after concatenation
    x = hybrid_block(
        x,
        filters_per_branch=16,
        output_filters=48,
        dilation_rate=dilation_rate
    )

    x = layers.MaxPooling2D(pool_size=2)(x)

    # block 3
    # 32 filters per branch produce 96 feature maps after concatenation
    x = hybrid_block(
        x,
        filters_per_branch=32,
        output_filters=96,
        dilation_rate=dilation_rate
    )

    x = layers.MaxPooling2D(pool_size=2)(x)

    # block 4
    # 64 filters per branch produce 192 feature maps after concatenation
    x = hybrid_block(
        x,
        filters_per_branch=64,
        output_filters=192,
        dilation_rate=dilation_rate
    )

    # converts the final feature maps into one feature vector
    x = layers.GlobalAveragePooling2D()(x)

    # helps reduce overfitting before the final classification layer
    x = layers.Dropout(0.3)(x)

    # predicts one of the 10 CIFAR-10 classes
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="HybridCNN"
    )