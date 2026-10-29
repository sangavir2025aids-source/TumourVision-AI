import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from data_loader import val_dataset

MODEL_PATH = "models/segmentation_model/best_unet_tumour_focused.keras"

print("Loading improved tumour-focused model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Improved model loaded successfully!")

selected_image = None
selected_mask = None
selected_prediction = None

print("\n========== SEARCHING FOR TUMOUR IMAGE ==========")

for batch_number, (images, masks) in enumerate(val_dataset):

    print(f"Checking validation batch {batch_number + 1}...")

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_masks = np.argmax(
        predictions,
        axis=-1
    )

    for i in range(len(masks)):

        mask = masks[i].numpy().squeeze()

        if np.any(mask == 2):

            selected_image = images[i].numpy().squeeze()
            selected_mask = mask
            selected_prediction = predicted_masks[i]

            print("\nTumour image found!")
            print("Batch:", batch_number + 1)
            print("Image number:", i + 1)

            break

    if selected_image is not None:
        break


if selected_image is None:

    print("\nNo tumour image found in validation dataset.")

else:

    true_tumour = selected_mask == 2
    predicted_tumour = selected_prediction == 2

    actual_pixels = np.sum(true_tumour)
    predicted_pixels = np.sum(predicted_tumour)

    intersection = np.logical_and(
        true_tumour,
        predicted_tumour
    ).sum()

    union = np.logical_or(
        true_tumour,
        predicted_tumour
    ).sum()

    if actual_pixels + predicted_pixels > 0:
        dice = (
            2 * intersection
        ) / (
            actual_pixels + predicted_pixels
        )
    else:
        dice = 1.0

    if union > 0:
        iou = intersection / union
    else:
        iou = 1.0

    print("\nActual tumour pixels:", actual_pixels)
    print("Predicted tumour pixels:", predicted_pixels)
    print("Intersection pixels:", intersection)
    print("Dice Score:", round(dice, 4))
    print("IoU Score:", round(iou, 4))

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(
        selected_image,
        cmap="gray"
    )
    plt.title("Original CT")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(
        true_tumour,
        cmap="gray"
    )
    plt.title("Actual Tumour")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(
        predicted_tumour,
        cmap="gray"
    )
    plt.title("AI Prediction")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(
        selected_image,
        cmap="gray"
    )
    plt.imshow(
        predicted_tumour,
        alpha=0.5
    )
    plt.title("AI Tumour Overlay")
    plt.axis("off")

    plt.tight_layout()

    output_path = "outputs/tumour_prediction.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print("\n========== VISUALIZATION COMPLETE ==========")
    print("Visualization saved successfully!")
    print("Saved at:", output_path)

    plt.show()