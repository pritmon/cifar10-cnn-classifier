"""
train.py
--------
Builds, trains, and saves a simple CNN that classifies CIFAR-10 images
into 10 categories (airplane, automobile, bird, cat, deer, dog, frog,
horse, ship, truck).

Run it with:
    python train.py

After training finishes, the model is saved to model/cifar10_model.keras
so that predict.py can load it later.
"""

# --- Imports: the tools we need ---
import os                                      # to create folders / build file paths
import tensorflow as tf                        # the deep-learning engine (does the heavy maths)
from tensorflow.keras import layers, models    # ready-made building blocks for the network

# -----------------------------------------------------------------------------
# 1. LOAD THE DATA
# -----------------------------------------------------------------------------
# CIFAR-10 is a built-in Keras dataset, so the first run will download it
# (~170 MB) automatically. It gives us:
#   x_train / x_test -> the images, shape (N, 32, 32, 3) -> 32x32 color pixels
#   y_train / y_test -> the correct label for each image (a number 0-9)
print("Loading CIFAR-10 dataset (downloads on first run)...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(f"Training images: {x_train.shape[0]}, Test images: {x_test.shape[0]}")

# -----------------------------------------------------------------------------
# 2. NORMALIZE THE PIXEL VALUES
# -----------------------------------------------------------------------------
# Pixel values come in as integers 0-255. Neural networks train better and
# faster when inputs are small floating-point numbers, so we scale everything
# into the 0.0-1.0 range by dividing by 255.0.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# -----------------------------------------------------------------------------
# 3. BUILD THE CNN MODEL (Keras Sequential API)
# -----------------------------------------------------------------------------
# We stack layers one after another. The pattern is a classic 3-block CNN:
# each "block" is a Conv2D (finds patterns) followed by MaxPooling2D (shrinks
# the image so deeper layers see bigger-picture features and run faster).
model = models.Sequential([
    # Tell the model what shape each input image is: 32x32 pixels, 3 color channels.
    layers.Input(shape=(32, 32, 3)),

    # --- Block 1 ---
    # Conv2D learns 32 small filters that detect simple features like edges
    # and colors. 'relu' lets the network model non-linear patterns.
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    # MaxPooling2D halves the width/height (32x32 -> 16x16), keeping the
    # strongest signal in each 2x2 area. This cuts computation and adds
    # a little robustness to where features appear.
    layers.MaxPooling2D((2, 2)),

    # --- Block 2 ---
    # More filters (64) so the network can learn more complex shapes built
    # from the simple edges found in block 1.
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),  # 16x16 -> 8x8

    # --- Block 3 ---
    # Even more filters (128) for high-level features like textures/parts.
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),  # 8x8 -> 4x4

    # --- Classifier head ---
    # Flatten turns the 3D feature maps (4x4x128) into a flat 1D list of
    # numbers so we can feed them into regular (Dense) layers.
    layers.Flatten(),
    # A Dense layer with 128 neurons combines those features to start
    # forming a decision about what the image shows.
    layers.Dense(128, activation="relu"),
    # Dropout randomly "turns off" 30% of neurons during training. This
    # prevents the model from over-memorizing the training data (overfitting)
    # and helps it generalize to new images.
    layers.Dropout(0.3),
    # Final output layer: 10 neurons, one per class. 'softmax' turns the
    # outputs into probabilities that add up to 1 (e.g. 0.9 = 90% sure).
    layers.Dense(10, activation="softmax"),
])

# Show a text summary of the layers and parameter counts.
model.summary()

# -----------------------------------------------------------------------------
# 4. COMPILE THE MODEL
# -----------------------------------------------------------------------------
# Before training we choose:
#   - optimizer: 'adam' is a reliable, beginner-friendly default that adjusts
#     the learning rate automatically.
#   - loss: 'sparse_categorical_crossentropy' is the right loss when labels
#     are plain integers (0-9) and outputs are softmax probabilities.
#   - metrics: track 'accuracy' so we can read how often predictions are right.
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# -----------------------------------------------------------------------------
# 5. TRAIN THE MODEL
# -----------------------------------------------------------------------------
# fit() does the actual learning. Keras automatically prints the training
# accuracy and validation accuracy after every epoch.
#   - epochs=15: go through the whole training set 15 times.
#   - batch_size=64: update the model after every 64 images.
#   - validation_split=0.2: hold back 20% of the training data to check how
#     well the model does on images it is NOT learning from.
print("\nStarting training...\n")
history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.2,
)

# -----------------------------------------------------------------------------
# 6. EVALUATE ON THE TEST SET
# -----------------------------------------------------------------------------
# The test set is data the model has never seen at all, so this is the most
# honest measure of real-world performance.
print("\nEvaluating on the held-out test set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# -----------------------------------------------------------------------------
# 7. SAVE THE TRAINED MODEL
# -----------------------------------------------------------------------------
# We save in the modern .keras format. predict.py will load this exact file.
os.makedirs("model", exist_ok=True)
save_path = os.path.join("model", "cifar10_model.keras")
model.save(save_path)
print(f"\nModel saved to {save_path}")
print("You can now run: python predict.py <path_to_image>")
