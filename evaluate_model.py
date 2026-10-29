import tensorflow as tf
import numpy as np

from data_loader import val_dataset
MODEL_PATH = "models/segmentation_model/best_unet_weighted.keras"

print("Loading improved tumour-focused model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Improved model loaded successfully!")


def calculate_dice(y_true, y_pred):

    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.squeeze(y_true, axis=-1)

    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.int32)

    true_tumour = tf.cast(y_true == 2, tf.float32)
    pred_tumour = tf.cast(y_pred == 2, tf.float32)

    intersection = tf.reduce_sum(
        true_tumour * pred_tumour
    )

    total = (
        tf.reduce_sum(true_tumour)
        + tf.reduce_sum(pred_tumour)
    )

    return (
        2.0 * intersection + 1e-6
    ) / (
        total + 1e-6
    )


def calculate_iou(y_true, y_pred):

    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.squeeze(y_true, axis=-1)

    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.int32)

    true_tumour = tf.cast(y_true == 2, tf.float32)
    pred_tumour = tf.cast(y_pred == 2, tf.float32)

    intersection = tf.reduce_sum(
        true_tumour * pred_tumour
    )

    union = (
        tf.reduce_sum(true_tumour)
        + tf.reduce_sum(pred_tumour)
        - intersection
    )

    return (
        intersection + 1e-6
    ) / (
        union + 1e-6
    )


dice_scores = []
iou_scores = []

print("\n========== STARTING IMPROVED MODEL EVALUATION ==========")
print("Evaluating first 100 validation images...\n")


for batch_number, (images, masks) in enumerate(val_dataset):

    if batch_number >= 13:
        break

    predictions = model.predict(
        images,
        verbose=0
    )

    dice = calculate_dice(
        masks,
        predictions
    )

    iou = calculate_iou(
        masks,
        predictions
    )

    dice_scores.append(float(dice))
    iou_scores.append(float(iou))

    print(
        f"Batch {batch_number + 1}/13 completed | "
        f"Dice: {float(dice):.4f} | "
        f"IoU: {float(iou):.4f}"
    )


mean_dice = np.mean(dice_scores)
mean_iou = np.mean(iou_scores)


print("\n========== EVALUATION COMPLETE ==========")

print(
    "Improved Tumour Dice Score:",
    round(mean_dice, 4)
)

print(
    "Improved Tumour IoU Score:",
    round(mean_iou, 4)
)

print("\n========== COMPARISON ==========")

print("Old model Dice: 0.0835")
print("Old model IoU : 0.0804")

print(
    "Improved model Dice:",
    round(mean_dice, 4)
)

print(
    "Improved model IoU :",
    round(mean_iou, 4)
)