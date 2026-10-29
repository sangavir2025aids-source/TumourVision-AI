from pathlib import Path
from PIL import Image
import shutil

BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "dataset" / "processed" / "train_images"
MASK_DIR = BASE_DIR / "dataset" / "processed" / "train_masks"

TRAIN_IMAGE_DIR = BASE_DIR / "dataset" / "processed" / "train_256" / "images"
TRAIN_MASK_DIR = BASE_DIR / "dataset" / "processed" / "train_256" / "masks"

VAL_IMAGE_DIR = BASE_DIR / "dataset" / "processed" / "val_256" / "images"
VAL_MASK_DIR = BASE_DIR / "dataset" / "processed" / "val_256" / "masks"

# Create folders
for folder in [
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VAL_IMAGE_DIR,
    VAL_MASK_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

# Get all CT images
image_files = sorted(
    IMAGE_DIR.glob("*.png"),
    key=lambda x: (
        int(x.stem.split("-")[1]),
        int(x.stem.split("-")[3])
    )
)

print("Total image slices:", len(image_files))

# Use volume number for splitting
volume_numbers = sorted(
    set(int(f.stem.split("-")[1]) for f in image_files)
)

print("Total volumes:", len(volume_numbers))

# 80% volumes for training
split_index = int(len(volume_numbers) * 0.8)

train_volumes = set(volume_numbers[:split_index])
val_volumes = set(volume_numbers[split_index:])

print("Training volumes:", len(train_volumes))
print("Validation volumes:", len(val_volumes))

train_count = 0
val_count = 0

for image_path in image_files:

    parts = image_path.stem.split("-")

    volume_number = int(parts[1])
    slice_number = int(parts[3])

    # Matching mask
    mask_path = MASK_DIR / (
        f"segmentation-{volume_number}-slice-{slice_number:03d}.png"
    )

    if not mask_path.exists():
        print("Missing mask:", mask_path.name)
        continue

    # Resize CT
    image = Image.open(image_path).convert("L")
    image = image.resize(
        (256, 256),
        Image.Resampling.BILINEAR
    )

    # Resize mask
    mask = Image.open(mask_path)
    mask = mask.resize(
        (256, 256),
        Image.Resampling.NEAREST
    )

    # Training set
    if volume_number in train_volumes:

        image.save(
            TRAIN_IMAGE_DIR / image_path.name
        )

        mask.save(
            TRAIN_MASK_DIR / mask_path.name
        )

        train_count += 1

    # Validation set
    else:

        image.save(
            VAL_IMAGE_DIR / image_path.name
        )

        mask.save(
            VAL_MASK_DIR / mask_path.name
        )

        val_count += 1

print("\n========== SPLIT COMPLETE ==========")
print("Training slices:", train_count)
print("Validation slices:", val_count)
print("Total:", train_count + val_count)