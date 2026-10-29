import tensorflow as tf
from pathlib import Path

IMAGE_DIR = Path("dataset/processed/train_256/images")
MASK_DIR = Path("dataset/processed/train_256/masks")

SMALL_TUMOUR_MAX_PIXELS = 500

small_images = []
small_masks = []

print("Searching for small-tumour training slices...")


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

    mask_array = mask.numpy()

    tumour_pixels = (mask_array == 2).sum()

    if 0 < tumour_pixels < SMALL_TUMOUR_MAX_PIXELS:

        small_images.append(str(image_path))
        small_masks.append(str(mask_path))


print("Small-tumour images:", len(small_images))
print("Small-tumour masks :", len(small_masks))


def load_image_mask(image_path, mask_path):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.cast(mask, tf.int32)

    return image, mask


small_tumour_dataset = tf.data.Dataset.from_tensor_slices(
    (small_images, small_masks)
)

small_tumour_dataset = small_tumour_dataset.map(
    load_image_mask,
    num_parallel_calls=tf.data.AUTOTUNE
)

small_tumour_dataset = small_tumour_dataset.shuffle(
    buffer_size=1000
)

small_tumour_dataset = small_tumour_dataset.batch(8)

small_tumour_dataset = small_tumour_dataset.prefetch(
    tf.data.AUTOTUNE
)


print("Small-tumour dataset created successfully!")

# Verify one batch
for images, masks in small_tumour_dataset.take(1):

    print("\n========== DATASET CHECK ==========")

    print("Image shape:", images.shape)
    print("Mask shape :", masks.shape)

    print(
        "Image min:",
        float(tf.reduce_min(images))
    )

    print(
        "Image max:",
        float(tf.reduce_max(images))
    )

    print(
        "Mask labels:",
        tf.unique(tf.reshape(masks, [-1])).y.numpy()
    )

    tumour_pixels = tf.reduce_sum(
        tf.cast(masks == 2, tf.int32)
    )

    print(
        "Tumour pixels in batch:",
        int(tumour_pixels)
    )

    print("\nSmall-tumour dataset verification successful!")