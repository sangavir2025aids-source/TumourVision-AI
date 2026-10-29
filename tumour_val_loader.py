import tensorflow as tf
from pathlib import Path

IMG_SIZE = 256
BATCH_SIZE = 8

IMAGE_DIR = Path("dataset/processed/val_256/images")
MASK_DIR = Path("dataset/processed/val_256/masks")

image_paths = sorted(IMAGE_DIR.glob("*.png"))

tumour_images = []
tumour_masks = []

for image_path in image_paths:

    mask_name = image_path.name.replace(
        "volume-",
        "segmentation-"
    )

    mask_path = MASK_DIR / mask_name

    if not mask_path.exists():
        continue

    mask = tf.io.read_file(str(mask_path))
    mask = tf.image.decode_png(mask, channels=1)

    # Check whether this slice contains tumour label 2
    if tf.reduce_any(tf.equal(mask, 2)):
        tumour_images.append(str(image_path))
        tumour_masks.append(str(mask_path))

print("Tumour validation images:", len(tumour_images))
print("Tumour validation masks :", len(tumour_masks))


def load_image_mask(image_path, mask_path):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.cast(mask, tf.int32)

    return image, mask


tumour_val_dataset = tf.data.Dataset.from_tensor_slices(
    (tumour_images, tumour_masks)
)

tumour_val_dataset = tumour_val_dataset.map(
    load_image_mask,
    num_parallel_calls=tf.data.AUTOTUNE
)

tumour_val_dataset = tumour_val_dataset.batch(BATCH_SIZE)
tumour_val_dataset = tumour_val_dataset.prefetch(tf.data.AUTOTUNE)

print("Tumour validation dataset created successfully!")