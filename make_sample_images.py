"""
make_sample_images.py
---------------------
Helper script that saves a few real CIFAR-10 test images into the
sample_images/ folder so you have something to run predict.py on.

Run it once (after installing requirements):
    python make_sample_images.py

It saves one example of several classes, e.g. sample_images/cat.png.
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Load the test set (downloads CIFAR-10 on first run if not already cached).
print("Loading CIFAR-10 to grab a few sample images...")
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

os.makedirs("sample_images", exist_ok=True)

# Grab the first test image we find for each of these 5 classes.
wanted = ["cat", "dog", "airplane", "ship", "frog"]
saved = set()

for img, label in zip(x_test, y_test):
    name = CLASS_NAMES[int(label[0])]
    if name in wanted and name not in saved:
        # array_to_img turns the raw 32x32x3 pixel array into a PIL image
        # we can save straight to disk as a PNG.
        out_path = os.path.join("sample_images", f"{name}.png")
        array_to_img(img).save(out_path)
        print(f"Saved {out_path}")
        saved.add(name)
    if len(saved) == len(wanted):
        break

print("Done. Try: python predict.py sample_images/cat.png")
