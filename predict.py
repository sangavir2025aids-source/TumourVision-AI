import os
import cv2
import numpy as np
import tensorflow as tf

# --------------------------------------------------
# MODEL
# --------------------------------------------------

MODEL_PATH = "models/segmentation_model/best_unet_weighted.keras"

print("Loading TumourVision AI model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)
print("Model loaded successfully!")


# --------------------------------------------------
# PREDICT TUMOUR
# --------------------------------------------------

def predict_tumour(image_path):

    # Read image
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if original is None:
        raise ValueError("Unable to read CT image.")

    original_height, original_width = original.shape

    # --------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------

    resized = cv2.resize(
        original,
        (256, 256),
        interpolation=cv2.INTER_LINEAR
    )

    image = resized.astype(np.float32) / 255.0

    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    # --------------------------------------------------
    # AI PREDICTION
    # --------------------------------------------------

    prediction = model.predict(image, verbose=0)

    predicted_mask = np.argmax(
        prediction[0],
        axis=-1
    ).astype(np.uint8)

    # Resize prediction to original image size
    predicted_mask = cv2.resize(
        predicted_mask,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    )

    # --------------------------------------------------
    # LIVER MASK
    # Class 1 = Liver
    # Class 2 = Tumour
    # --------------------------------------------------

    liver_mask = (predicted_mask == 1).astype(np.uint8)
    tumour_mask = (predicted_mask == 2).astype(np.uint8)

    tumour_pixels = int(np.sum(tumour_mask))

    # --------------------------------------------------
    # DEFAULT VALUES
    # --------------------------------------------------

    small_count = 0
    medium_count = 0
    large_count = 0
    detected_regions = 0

    tumour_location = "Not detected"

    # --------------------------------------------------
    # FIND TUMOUR LOCATION
    # --------------------------------------------------

    if tumour_pixels > 0:

        # Find tumour connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            tumour_mask,
            connectivity=8
        )

        # Store valid tumour regions
        tumour_centroids = []

        for i in range(1, num_labels):

            area = int(stats[i, cv2.CC_STAT_AREA])

            if area < 20:
                continue

            detected_regions += 1

            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])

            cx, cy = centroids[i]

            tumour_centroids.append((cx, cy))

            # --------------------------------------------------
            # TUMOUR SIZE CATEGORY
            # --------------------------------------------------

            if area < 500:
                small_count += 1
                border_color = (16, 185, 129)       # Green

            elif area <= 2000:
                medium_count += 1
                border_color = (245, 158, 11)       # Orange

            else:
                large_count += 1
                border_color = (239, 68, 68)         # Red

        # --------------------------------------------------
        # FIND OVERALL TUMOUR CENTROID
        # --------------------------------------------------

        tumour_points = np.column_stack(
            np.where(tumour_mask > 0)
        )

        tumour_y = float(np.mean(tumour_points[:, 0]))
        tumour_x = float(np.mean(tumour_points[:, 1]))

        # --------------------------------------------------
        # FIND LIVER BOUNDING BOX
        # --------------------------------------------------

        liver_points = np.column_stack(
            np.where(liver_mask > 0)
        )

        if len(liver_points) > 0:

            liver_y_min = np.min(liver_points[:, 0])
            liver_y_max = np.max(liver_points[:, 0])

            liver_x_min = np.min(liver_points[:, 1])
            liver_x_max = np.max(liver_points[:, 1])

            liver_width = liver_x_max - liver_x_min

            # --------------------------------------------------
            # APPROXIMATE LIVER LOBE
            #
            # NOTE:
            # This is a project/demo estimation.
            # It is NOT clinical anatomical segmentation.
            # --------------------------------------------------

            relative_x = (
                (tumour_x - liver_x_min) /
                max(liver_width, 1)
            )

            # CT images are commonly displayed using
            # radiological orientation.
            #
            # Left side of image ≈ patient's Right side
            # Right side of image ≈ patient's Left side

            if relative_x < 0.40:

                tumour_location = "Right Lobe"

            elif relative_x > 0.60:

                tumour_location = "Left Lobe"

            else:

                tumour_location = "Central Region"

        else:

            tumour_location = "Liver region not detected"

    # --------------------------------------------------
    # OVERALL SIZE
    # --------------------------------------------------

    if tumour_pixels == 0:

        tumour_size = "No Tumour"

    elif large_count > 0:

        tumour_size = "Large"

    elif medium_count > 0:

        tumour_size = "Medium"

    else:

        tumour_size = "Small"

    # --------------------------------------------------
    # CREATE HIGHLIGHT IMAGE
    # --------------------------------------------------

    output_image = cv2.cvtColor(
        original,
        cv2.COLOR_GRAY2BGR
    )

    # --------------------------------------------------
    # DRAW EACH TUMOUR REGION
    # --------------------------------------------------

    if tumour_pixels > 0:

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            tumour_mask,
            connectivity=8
        )

        for i in range(1, num_labels):

            area = int(stats[i, cv2.CC_STAT_AREA])

            if area < 20:
                continue

            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])

            # Size-based colours
            if area < 500:

                border_color = (16, 185, 129)      # Green
                label = "Small"

            elif area <= 2000:

                border_color = (245, 158, 11)      # Orange
                label = "Medium"

            else:

                border_color = (239, 68, 68)        # Red
                label = "Large"

            # Draw bounding box
            cv2.rectangle(
                output_image,
                (x, y),
                (x + w, y + h),
                border_color,
                3
            )

            # Draw contour
            region_mask = (
                (labels == i).astype(np.uint8) * 255
            )

            contours, _ = cv2.findContours(
                region_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(
                output_image,
                contours,
                -1,
                border_color,
                3
            )

            # Label
            cv2.putText(
                output_image,
                label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                border_color,
                2
            )

    # --------------------------------------------------
    # ADD LEGEND
    # --------------------------------------------------

    legend_x = 20
    legend_y = original_height - 110

    cv2.putText(
        output_image,
        "Tumour Size:",
        (legend_x, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output_image,
        "Green = Small",
        (legend_x, legend_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (16, 185, 129),
        2
    )

    cv2.putText(
        output_image,
        "Orange = Medium",
        (legend_x, legend_y + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (245, 158, 11),
        2
    )

    cv2.putText(
        output_image,
        "Red = Large",
        (legend_x, legend_y + 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (239, 68, 68),
        2
    )

    # --------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------

    output_folder = "outputs"

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    output_path = os.path.join(
        output_folder,
        "tumour_highlighted.png"
    )

    cv2.imwrite(
        output_path,
        output_image
    )

    # --------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------

    return (
        original,
        predicted_mask,
        tumour_pixels,
        output_path,
        small_count,
        medium_count,
        large_count,
        detected_regions,
        tumour_location
    )