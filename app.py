"""
app.py
------
A small web app (built with Gradio) that puts the trained CNN online.
Visitors drag in any image and instantly see the model's top guesses.

Run it locally with:
    python app.py
then open the http://127.0.0.1:7860 link it prints.

This same file is what runs on Hugging Face Spaces when deployed.
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

# Load the trained model once, when the app starts (not on every request).
# We use the augmented model because it is the more accurate of the two.
MODEL_PATH = "model/cifar10_model_augmented.keras"
model = tf.keras.models.load_model(MODEL_PATH)


def classify(img: Image.Image):
    """Take an uploaded image and return each class with its probability."""
    # The model only understands 32x32 colour images, so prepare the input:
    img = img.convert("RGB").resize((32, 32))      # force 3 colour channels, resize
    arr = np.array(img).astype("float32") / 255.0  # normalise pixels to 0-1 (as in training)
    arr = np.expand_dims(arr, axis=0)              # wrap as a "batch of 1": (1, 32, 32, 3)

    # Run the model and pull out the 10 probabilities for this one image.
    probs = model.predict(arr, verbose=0)[0]

    # Gradio's Label component shows a nice bar chart from a {name: score} dict.
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


# --- Build the web page ---
demo = gr.Interface(
    fn=classify,                                       # the function each upload runs
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=3, label="Top 3 guesses"),
    title="🖼️ CIFAR-10 CNN Image Classifier",
    description=(
        "Drag in a photo and the model sorts it into one of 10 classes: "
        "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. "
        "Tip: it was trained on tiny 32×32 images, so simple, centred subjects work best."
    ),
    examples=[
        "sample_images/cat.png",
        "sample_images/dog.png",
        "sample_images/ship.png",
        "sample_images/airplane.png",
        "sample_images/frog.png",
    ],
    cache_examples=False,   # compute example results on click, not at startup
)

# Start the local web server when you run `python app.py`.
if __name__ == "__main__":
    demo.launch()
