import tensorflow as tf


def tumour_weighted_loss(y_true, y_pred):

    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.squeeze(y_true, axis=-1)

    # Sparse categorical cross-entropy
    ce = tf.keras.losses.sparse_categorical_crossentropy(
        y_true,
        y_pred
    )

    # Give tumour pixels higher importance
    weights = tf.where(
        y_true == 2,
        6.0,
        tf.where(
            y_true == 1,
            1.0,
            0.2
        )
    )

    weighted_ce = ce * weights
    weighted_ce = tf.reduce_mean(weighted_ce)

    # Tumour Dice
    tumour_true = tf.cast(
        y_true == 2,
        tf.float32
    )

    tumour_pred = y_pred[:, :, :, 2]

    tumour_true = tf.reshape(
        tumour_true,
        [tf.shape(tumour_true)[0], -1]
    )

    tumour_pred = tf.reshape(
        tumour_pred,
        [tf.shape(tumour_pred)[0], -1]
    )

    intersection = tf.reduce_sum(
        tumour_true * tumour_pred,
        axis=1
    )

    denominator = tf.reduce_sum(
        tumour_true + tumour_pred,
        axis=1
    )

    smooth = 1e-6

    dice = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    dice_loss = 1.0 - tf.reduce_mean(dice)

    # Final loss
    total_loss = weighted_ce + 2.0 * dice_loss

    return total_loss


print("Tumour-weighted Dice loss loaded successfully!")