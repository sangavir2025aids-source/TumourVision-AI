from pathlib import Path
import tensorflow as tf

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Dataset folders
TRAIN_IMAGE_DIR = BASE_DIR / "dataset" / "processed" / "train_256" / "images"
TRAIN_MASK_DIR = BASE_DIR / "dataset" / "processed" / "train_256" / "masks"

VAL_IMAGE_DIR = BASE_DIR / "dataset" / "processed" / "val_256" / "images"
VAL_MASK_DIR = BASE_DIR / "dataset" / "processed" / "val_256" / "masks"


IMG_SIZE = 256
BATCH_SIZE = 8


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image


def load_mask(mask_path):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)

    # Convert mask to integer class labels
    mask = tf.cast(mask, tf.int32)

    return mask


def load_pair(image_path, mask_path):
    image = load_image(image_path)
    mask = load_mask(mask_path)

    return image, mask


# Get training files
train_images = sorted(
    [str(x) for x in TRAIN_IMAGE_DIR.glob("*.png")]
)

train_masks = sorted(
    [str(x) for x in TRAIN_MASK_DIR.glob("*.png")]
)

# Get validation files
val_images = sorted(
    [str(x) for x in VAL_IMAGE_DIR.glob("*.png")]
)

val_masks = sorted(
    [str(x) for x in VAL_MASK_DIR.glob("*.png")]
)


print("Training images:", len(train_images))
print("Training masks:", len(train_masks))

print("Validation images:", len(val_images))
print("Validation masks:", len(val_masks))


# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_images, train_masks)
)

train_dataset = train_dataset.map(
    load_pair,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.shuffle(500)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)


val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_images, val_masks)
)

val_dataset = val_dataset.map(
    load_pair,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)


# Check one batch
for images, masks in train_dataset.take(1):

    print("\nOne training batch:")
    print("Image shape:", images.shape)
    print("Mask shape:", masks.shape)
    print("Image minimum:", tf.reduce_min(images).numpy())
    print("Image maximum:", tf.reduce_max(images).numpy())
    print("Mask labels:",
          tf.unique(tf.reshape(masks, [-1])).y.numpy())