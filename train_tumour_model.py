import tensorflow as tf
import os

from unet_model import build_unet
from tumour_data_loader import tumour_dataset
from data_loader import val_dataset
from losses import combined_loss


MODEL_DIR = "models/segmentation_model"

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_unet_tversky.keras"
)


print("\n========== BUILDING MODEL ==========")

model = build_unet(
    input_shape=(256, 256, 1),
    num_classes=3
)

print("Model created successfully!")


print("\n========== COMPILING MODEL ==========")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss=combined_loss,
    metrics=["accuracy"]
)

print("Model compiled successfully!")


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)


print("\n========== STARTING TRAINING ==========")

history = model.fit(
    tumour_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)


print("\n========== TRAINING COMPLETE ==========")

print("Best model saved at:")
print(MODEL_PATH)