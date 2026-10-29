import tensorflow as tf
from tensorflow.keras import layers, Model


def conv_block(x, filters):

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    return x


def build_unet(
    input_shape=(256, 256, 1),
    num_classes=3
):

    inputs = layers.Input(
        shape=input_shape
    )

    # Encoder
    c1 = conv_block(inputs, 16)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 32)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 64)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 128)
    p4 = layers.MaxPooling2D()(c4)

    # Bottleneck
    c5 = conv_block(p4, 256)

    # Decoder
    u6 = layers.UpSampling2D()(c5)
    u6 = layers.Concatenate()([u6, c4])
    c6 = conv_block(u6, 128)

    u7 = layers.UpSampling2D()(c6)
    u7 = layers.Concatenate()([u7, c3])
    c7 = conv_block(u7, 64)

    u8 = layers.UpSampling2D()(c7)
    u8 = layers.Concatenate()([u8, c2])
    c8 = conv_block(u8, 32)

    u9 = layers.UpSampling2D()(c8)
    u9 = layers.Concatenate()([u9, c1])
    c9 = conv_block(u9, 16)

    # Output
    outputs = layers.Conv2D(
        num_classes,
        1,
        activation="softmax"
    )(c9)

    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    return model


if __name__ == "__main__":

    model = build_unet()

    print("Smaller U-Net created successfully!")

    print("Input shape:", model.input_shape)

    print("Output shape:", model.output_shape)

    print(
        "Total parameters:",
        model.count_params()
    )