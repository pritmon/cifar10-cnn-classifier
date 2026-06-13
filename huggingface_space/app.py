"""
app.py  (Hugging Face Spaces version)
-------------------------------------
The web app that runs on Hugging Face Spaces. Same logic as the project's
root app.py, but every file sits in one flat folder (no subfolders), which
is the simplest layout to upload to a Space.
"""

# --- Imports: the tools we need ---
import gradio as gr          # builds the drag-and-drop web interface
import numpy as np           # fast maths on the image numbers
import tensorflow as tf      # loads and runs the trained model
from PIL import Image        # represents the uploaded picture

# The 10 categories, in the exact order the model was trained on.
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# The trained model lives in the same folder on the Space (flat layout).
MODEL_PATH = "cifar10_model_augmented.keras"
model = tf.keras.models.load_model(MODEL_PATH)


def classify(img: Image.Image):
    """Take an uploaded image and return each class with its probability."""
    img = img.convert("RGB").resize((32, 32))      # 3 colour channels, resize to 32x32
    arr = np.array(img).astype("float32") / 255.0  # normalise pixels to 0-1 (as in training)
    arr = np.expand_dims(arr, axis=0)              # wrap as a batch of 1: (1, 32, 32, 3)
    probs = model.predict(arr, verbose=0)[0]       # the 10 probabilities for this image
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


# --- Build the web page ---
demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=3, label="Top 3 guesses"),
    title="🖼️ CIFAR-10 CNN Image Classifier",
    description=(
        "Drag in a photo and the model sorts it into one of 10 classes: "
        "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. "
        "Tip: it was trained on tiny 32×32 images, so simple, centred subjects work best."
    ),
    examples=["cat.png", "dog.png", "ship.png", "airplane.png", "frog.png"],
)

if __name__ == "__main__":
    demo.launch()
