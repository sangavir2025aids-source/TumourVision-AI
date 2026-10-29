import tensorflow as tf
import numpy as np

from tumour_val_loader import tumour_val_dataset

MODEL_PATH = "models/segmentation_model/best_unet_weighted.keras"

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")


dice_scores = []
iou_scores = []

print("\nEvaluating tumour-containing validation slices...")

for images, masks in tumour_val_dataset:

    predictions = model.predict(images, verbose=0)

    predicted_mask = np.argmax(predictions, axis=-1)
    true_mask = masks.numpy().squeeze(-1)

    # Tumour = label 2
    true_tumour = (true_mask == 2)
    predicted_tumour = (predicted_mask == 2)

    intersection = np.logical_and(
        true_tumour,
        predicted_tumour
    ).sum(axis=(1, 2))

    true_area = true_tumour.sum(axis=(1, 2))
    predicted_area = predicted_tumour.sum(axis=(1, 2))

    union = np.logical_or(
        true_tumour,
        predicted_tumour
    ).sum(axis=(1, 2))

    dice = (2 * intersection + 1e-6) / (
        true_area + predicted_area + 1e-6
    )

    iou = (intersection + 1e-6) / (
        union + 1e-6
    )

    dice_scores.extend(dice.tolist())
    iou_scores.extend(iou.tolist())


print("\n========== TUMOUR VALIDATION RESULTS ==========")

print(
    "Tumour Dice Score:",
    round(float(np.mean(dice_scores)), 4)
)

print(
    "Tumour IoU Score :",
    round(float(np.mean(iou_scores)), 4)
)

print("\nNumber of tumour-containing slices evaluated:",
      len(dice_scores))

print("\n========== EVALUATION COMPLETE ==========")