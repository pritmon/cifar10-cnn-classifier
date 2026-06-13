"""
predict.py
----------
Loads the trained CNN and predicts what is in a single image.

Run it with:
    python predict.py sample_images/cat.png

It prints the predicted class name and how confident the model is (as a %).
"""

import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

# -----------------------------------------------------------------------------
# 1. THE CLASS NAMES
# -----------------------------------------------------------------------------
# The model outputs a number 0-9. This list maps each number to a human-
# readable label. The ORDER matters: it must match CIFAR-10's label order.
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Where the trained model lives by default (created by train.py).
DEFAULT_MODEL_PATH = os.path.join("model", "cifar10_model.keras")


def main():
    # -------------------------------------------------------------------------
    # 2. READ THE IMAGE PATH FROM THE COMMAND LINE
    # -------------------------------------------------------------------------
    # The user must pass the image path. They can ALSO pass an optional second
    # argument to choose a different saved model (e.g. the augmented one).
    #   sys.argv[0] = script name
    #   sys.argv[1] = image path (required)
    #   sys.argv[2] = model path (optional, defaults to the original model)
    if len(sys.argv) not in (2, 3):
        print("Usage: python predict.py <path_to_image> [path_to_model]")
        print("Example: python predict.py sample_images/cat.png")
        print("Example: python predict.py sample_images/cat.png model/cifar10_model_augmented.keras")
        sys.exit(1)

    image_path = sys.argv[1]
    # Use the model the user named, or fall back to the default one.
    MODEL_PATH = sys.argv[2] if len(sys.argv) == 3 else DEFAULT_MODEL_PATH

    # Friendly checks so beginners get clear errors instead of crashes.
    if not os.path.exists(image_path):
        print(f"Error: image file not found: {image_path}")
        sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: model not found at {MODEL_PATH}")
        print("Train it first by running: python train.py")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 3. LOAD THE TRAINED MODEL
    # -------------------------------------------------------------------------
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # -------------------------------------------------------------------------
    # 4. LOAD AND PREPARE THE IMAGE
    # -------------------------------------------------------------------------
    # The model was trained on 32x32 color images, so we must resize whatever
    # image the user gives us to exactly 32x32 pixels.
    img = keras_image.load_img(image_path, target_size=(32, 32))

    # Convert the image into an array of numbers, shape (32, 32, 3).
    img_array = keras_image.img_to_array(img)

    # Normalize the same way we did in training (pixels 0-255 -> 0.0-1.0).
    # If we skip this, predictions will be unreliable.
    img_array = img_array / 255.0

    # The model expects a BATCH of images, not a single one. We add an extra
    # dimension at the front so the shape becomes (1, 32, 32, 3) -> a batch of 1.
    img_batch = np.expand_dims(img_array, axis=0)

    # -------------------------------------------------------------------------
    # 5. MAKE THE PREDICTION
    # -------------------------------------------------------------------------
    # predict() returns the 10 softmax probabilities for our single image.
    predictions = model.predict(img_batch, verbose=0)
    probabilities = predictions[0]  # pull the first (and only) result out

    # The predicted class is whichever probability is highest.
    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100.0

    # -------------------------------------------------------------------------
    # 6. SHOW THE RESULT
    # -------------------------------------------------------------------------
    print(f"\nPredicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    # Bonus: show the top 3 guesses so the user can see the runner-ups.
    print("\nTop 3 guesses:")
    top3 = np.argsort(probabilities)[::-1][:3]  # indices of 3 highest scores
    for rank, idx in enumerate(top3, start=1):
        print(f"  {rank}. {CLASS_NAMES[idx]:<12} {probabilities[idx] * 100:.2f}%")


if __name__ == "__main__":
    main()
