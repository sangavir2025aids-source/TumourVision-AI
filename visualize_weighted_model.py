import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_PATH = "models/segmentation_model/best_unet_weighted.keras"

IMAGE_DIR = Path("dataset/processed/val_256/images")
MASK_DIR = Path("dataset/processed/val_256/masks")

print("Loading weighted U-Net...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")

# Find the validation slice with the largest tumour
selected_image = None
selected_mask = None
largest_tumour_area = 0

for image_path in sorted(IMAGE_DIR.glob("*.png")):

    mask_name = image_path.name.replace(
        "volume-",
        "segmentation-"
    )

    mask_path = MASK_DIR / mask_name

    if not mask_path.exists():
        continue

    mask = tf.io.read_file(str(mask_path))
    mask = tf.image.decode_png(mask, channels=1)
    mask = mask.numpy()

    tumour_area = np.sum(mask == 2)

    if tumour_area > largest_tumour_area:
        largest_tumour_area = tumour_area
        selected_image = image_path
        selected_mask = mask_path


print("Selected image:", selected_image)
print("Largest tumour area:", largest_tumour_area)


print("Selected image:", selected_image)


# Load CT image
image = tf.io.read_file(str(selected_image))
image = tf.image.decode_png(image, channels=1)
image = tf.cast(image, tf.float32) / 255.0


# Load actual mask
mask = tf.io.read_file(str(selected_mask))
mask = tf.image.decode_png(mask, channels=1)
mask = mask.numpy().squeeze()


# Predict tumour
prediction = model.predict(
    tf.expand_dims(image, axis=0),
    verbose=0
)

predicted_mask = np.argmax(
    prediction[0],
    axis=-1
)


# Tumour = label 2
actual_tumour = mask == 2
predicted_tumour = predicted_mask == 2


# Calculate Dice
intersection = np.logical_and(
    actual_tumour,
    predicted_tumour
).sum()

actual_area = actual_tumour.sum()
predicted_area = predicted_tumour.sum()

dice = (
    2 * intersection + 1e-6
) / (
    actual_area + predicted_area + 1e-6
)


# Calculate IoU
union = np.logical_or(
    actual_tumour,
    predicted_tumour
).sum()

iou = (
    intersection + 1e-6
) / (
    union + 1e-6
)


print("\n========== RESULTS ==========")
print("Actual tumour pixels    :", actual_area)
print("Predicted tumour pixels :", predicted_area)
print("Intersection            :", intersection)
print("Dice                    :", round(float(dice), 4))
print("IoU                     :", round(float(iou), 4))


# Create output folder
Path("outputs").mkdir(exist_ok=True)


# Display results
plt.figure(figsize=(12, 4))


plt.subplot(1, 3, 1)
plt.imshow(
    image.numpy().squeeze(),
    cmap="gray"
)
plt.title("CT Scan")
plt.axis("off")


plt.subplot(1, 3, 2)
plt.imshow(
    actual_tumour,
    cmap="gray"
)
plt.title("Actual Tumour")
plt.axis("off")


plt.subplot(1, 3, 3)
plt.imshow(
    image.numpy().squeeze(),
    cmap="gray"
)

# Highlight predicted tumour
plt.imshow(
    predicted_tumour,
    cmap="Reds",
    alpha=0.5
)

plt.title(
    f"AI Prediction\nDice: {dice:.3f}"
)

plt.axis("off")


plt.tight_layout()


output_path = "outputs/weighted_model_prediction.png"

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print("\nVisualization saved to:")
print(output_path)

print("\n========== COMPLETE ==========")