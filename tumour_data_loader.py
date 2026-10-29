import tensorflow as tf
from pathlib import Path

IMG_SIZE = 256
BATCH_SIZE = 8

IMAGE_DIR = Path("dataset/processed/train_256/images")
MASK_DIR = Path("dataset/processed/train_256/masks")

tumour_images = []
tumour_masks = []

print("Searching for tumour-containing training slices...")

for mask_path in sorted(MASK_DIR.glob("*.png")):

    mask_data = tf.io.read_file(str(mask_path))

    mask = tf.image.decode_png(
        mask_data,
        channels=1
    )

    mask = tf.cast(mask, tf.int32)

    # Check whether this mask contains tumour label 2
    if bool(tf.reduce_any(mask == 2).numpy()):

        image_name = mask_path.name.replace(
            "segmentation-",
            "volume-"
        )

        image_path = IMAGE_DIR / image_name

        if image_path.exists():

            tumour_images.append(
                str(image_path)
            )

            tumour_masks.append(
                str(mask_path)
            )


print("Tumour-containing images:", len(tumour_images))
print("Tumour-containing masks:", len(tumour_masks))


def load_image_and_mask(image_path, mask_path):

    image_data = tf.io.read_file(image_path)

    image = tf.image.decode_png(
        image_data,
        channels=1
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0


    mask_data = tf.io.read_file(mask_path)

    mask = tf.image.decode_png(
        mask_data,
        channels=1
    )

    mask = tf.cast(
        mask,
        tf.int32
    )

    return image, mask


if len(tumour_images) == 0:

    print("\nERROR: No tumour-containing images were found.")
    print("Please check the image and mask filenames.")

else:

    tumour_dataset = tf.data.Dataset.from_tensor_slices(
        (tumour_images, tumour_masks)
    )

    tumour_dataset = tumour_dataset.shuffle(
        len(tumour_images),
        seed=42
    )

    tumour_dataset = tumour_dataset.map(
        load_image_and_mask,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    tumour_dataset = tumour_dataset.batch(
        BATCH_SIZE
    )

    tumour_dataset = tumour_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    print("Tumour-focused dataset created successfully!")