import tensorflow as tf
import numpy as np

from tumour_val_loader import tumour_val_dataset


MODEL_PATH = "models/segmentation_model/best_unet_weighted.keras"

print("Loading weighted U-Net...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")


small_dice = []
small_iou = []

medium_dice = []
medium_iou = []

large_dice = []
large_iou = []


print("\nEvaluating small, medium and large tumours...")


for images, masks in tumour_val_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_mask = np.argmax(
        predictions,
        axis=-1
    )

    true_mask = masks.numpy().squeeze(-1)


    # Tumour = label 2
    true_tumour = true_mask == 2
    predicted_tumour = predicted_mask == 2


    # Calculate values for every image in the batch
    for i in range(len(images)):

        actual = true_tumour[i]
        predicted = predicted_tumour[i]


        actual_area = np.sum(actual)

        intersection = np.logical_and(
            actual,
            predicted
        ).sum()

        union = np.logical_or(
            actual,
            predicted
        ).sum()

        predicted_area = np.sum(predicted)


        dice = (
            2 * intersection + 1e-6
        ) / (
            actual_area + predicted_area + 1e-6
        )


        iou = (
            intersection + 1e-6
        ) / (
            union + 1e-6
        )


        # Categorize tumour by actual area
        if actual_area < 500:

            small_dice.append(dice)
            small_iou.append(iou)

        elif actual_area <= 2000:

            medium_dice.append(dice)
            medium_iou.append(iou)

        else:

            large_dice.append(dice)
            large_iou.append(iou)


print("\n========== TUMOUR SIZE RESULTS ==========")


print("\nSMALL TUMOURS")
print("-------------------------")
print("Number of slices:", len(small_dice))

if len(small_dice) > 0:
    print(
        "Dice:",
        round(float(np.mean(small_dice)), 4)
    )
    print(
        "IoU :",
        round(float(np.mean(small_iou)), 4)
    )


print("\nMEDIUM TUMOURS")
print("-------------------------")
print("Number of slices:", len(medium_dice))

if len(medium_dice) > 0:
    print(
        "Dice:",
        round(float(np.mean(medium_dice)), 4)
    )
    print(
        "IoU :",
        round(float(np.mean(medium_iou)), 4)
    )


print("\nLARGE TUMOURS")
print("-------------------------")
print("Number of slices:", len(large_dice))

if len(large_dice) > 0:
    print(
        "Dice:",
        round(float(np.mean(large_dice)), 4)
    )
    print(
        "IoU :",
        round(float(np.mean(large_iou)), 4)
    )


print("\n========== EVALUATION COMPLETE ==========")