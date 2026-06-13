"""
train_augmented.py
------------------
Same CNN as train.py, but with DATA AUGMENTATION added.

The idea: instead of showing the model each photo exactly once, we show it
slightly changed copies every time -- flipped left-right, rotated a little,
zoomed in a bit. The model never sees the *exact* same image twice, so it
can't just memorize. It is forced to learn the real thing ("what a cat looks
like") instead of one specific cat photo. This usually helps the most on the
hard, easily-confused classes like cat vs dog.

Run it with:
    python train_augmented.py

It saves to model/cifar10_model_augmented.keras so your original
model/cifar10_model.keras is left untouched.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------------------------------------------------------
# 1. LOAD THE DATA (same as before)
# -----------------------------------------------------------------------------
# CIFAR-10 is already cached from the first run, so this is instant now.
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(f"Training images: {x_train.shape[0]}, Test images: {x_test.shape[0]}")

# -----------------------------------------------------------------------------
# 2. NORMALIZE THE PIXEL VALUES (same as before)
# -----------------------------------------------------------------------------
# Scale pixels from 0-255 down to 0.0-1.0 so the network trains nicely.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# -----------------------------------------------------------------------------
# 3. BUILD THE MODEL -- NOW WITH AUGMENTATION LAYERS AT THE FRONT
# -----------------------------------------------------------------------------
# These three augmentation layers are the only new part. They sit at the very
# start of the model and randomly tweak each image *during training only*.
# At prediction time they automatically do nothing, so predict.py is unaffected.
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),

    # --- Data augmentation (training-time only) ---
    # Randomly flip the image left-right. A cat facing left is still a cat,
    # so this doubles the variety for free.
    layers.RandomFlip("horizontal"),
    # Randomly rotate by up to ~10% of a full turn (about 36 degrees) so the
    # model learns animals/objects that aren't perfectly upright.
    layers.RandomRotation(0.1),
    # Randomly zoom in/out by up to 10% so it learns things at different sizes.
    layers.RandomZoom(0.1),

    # --- Block 1 --- (identical to train.py from here down)
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # --- Block 2 ---
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # --- Block 3 ---
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # --- Classifier head ---
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])

model.summary()

# -----------------------------------------------------------------------------
# 4. COMPILE (same settings as before)
# -----------------------------------------------------------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# -----------------------------------------------------------------------------
# 5. TRAIN -- FOR MORE EPOCHS THIS TIME
# -----------------------------------------------------------------------------
# Augmentation makes every epoch a little "harder" (the images keep changing),
# so the model learns more slowly but ends up stronger. Our first 30-epoch run
# was still improving when it stopped, so we give it 55 epochs to finish the
# job. Watch val_accuracy: if it stops climbing and starts dropping, that's the
# sign we've gone too far (overfitting) and could pull the number back down.
print("\nStarting training with augmentation (this takes longer)...\n")
history = model.fit(
    x_train, y_train,
    epochs=55,
    batch_size=64,
    validation_split=0.2,
)

# -----------------------------------------------------------------------------
# 6. EVALUATE ON THE TEST SET
# -----------------------------------------------------------------------------
print("\nEvaluating on the held-out test set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# -----------------------------------------------------------------------------
# 7. SAVE -- to a SEPARATE file so the original model is kept
# -----------------------------------------------------------------------------
os.makedirs("model", exist_ok=True)
save_path = os.path.join("model", "cifar10_model_augmented.keras")
model.save(save_path)
print(f"\nModel saved to {save_path}")
print("Predict with it like this:")
print(f"  python predict.py sample_images/cat.png {save_path}")
