# CIFAR-10 CNN Image Classifier

A TensorFlow / Keras project that trains a small
Convolutional Neural Network (CNN) to classify 32x32 color images into
**10 categories**:

`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

The code is heavily commented so you can learn what each piece does and why.

---

## Project structure

```
cifar10-cnn-classifier/
├── train.py               # build, train, and save the model
├── predict.py             # load the model and predict on one image
├── make_sample_images.py  # helper: saves a few CIFAR-10 images to predict on
├── model/                 # the trained model is saved here
├── sample_images/         # test images for predict.py
├── requirements.txt       # Python dependencies
└── README.md              # this file
```

---

## The model (a 3-block CNN)

Built with the Keras **Sequential** API:

```
Input (32x32x3)
 → Conv2D(32) → MaxPool      # block 1: learn simple edges/colors
 → Conv2D(64) → MaxPool      # block 2: learn shapes
 → Conv2D(128) → MaxPool     # block 3: learn complex features
 → Flatten
 → Dense(128, relu)
 → Dropout(0.3)              # reduce overfitting
 → Dense(10, softmax)        # output: probability for each class
```

Pixel values are normalized (divided by 255) so they fall in the 0–1 range,
which helps the network train faster and more reliably.

---

## Step 1 — Install

You need **Python 3.9–3.11** (TensorFlow 2.x supports these well).

It is best to use a virtual environment so packages stay isolated:

```bash
# from inside the cifar10-cnn-classifier/ folder

# create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows (PowerShell/CMD)

# install the dependencies
pip install -r requirements.txt
```

---

## Step 2 — Train the model

```bash
python train.py
```

What happens:

1. CIFAR-10 downloads automatically the first time (~170 MB).
2. The CNN trains for **15 epochs** with `batch_size=64` and a 20%
   validation split.
3. After every epoch you'll see the **training accuracy** and
   **validation accuracy** printed by Keras, for example:
   ```
   Epoch 5/15
   625/625 [====] - loss: 0.78 - accuracy: 0.72 - val_loss: 0.85 - val_accuracy: 0.70
   ```
4. The model is evaluated on the test set and saved to
   `model/cifar10_model.keras`.

> Training on a CPU takes a few minutes per epoch. A GPU is much faster but
> not required. Expect roughly **70–75% test accuracy** with this simple model.

---

## Step 3 — Get some images to predict on

Save a few real CIFAR-10 images into `sample_images/`:

```bash
python make_sample_images.py
```

This creates files like `sample_images/cat.png`, `sample_images/dog.png`, etc.
You can also use any of your own JPG/PNG images.

---

## Step 4 — Predict

Pass an image path to `predict.py`:

```bash
python predict.py sample_images/cat.png
```

Example output:

```
Predicted class: cat
Confidence: 87.34%

Top 3 guesses:
  1. cat          87.34%
  2. dog          8.12%
  3. deer         2.05%
```

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'tensorflow'`** — activate your
  virtual environment and re-run `pip install -r requirements.txt`.
- **`model not found`** when predicting — run `python train.py` first.
- **Slow training** — that's normal on a CPU; reduce `epochs` in `train.py`
  while experimenting.
- **Low confidence on your own photos** — CIFAR-10 images are tiny (32x32)
  and cover only 10 specific classes, so unusual photos may confuse the model.

---

## Bonus: the augmented model (`train_augmented.py`)

`train_augmented.py` is an upgraded version of the trainer. It adds **data
augmentation** — three layers (`RandomFlip`, `RandomRotation`, `RandomZoom`)
that randomly tweak each image during training so the model sees more variety
and stops memorizing. It also trains for more epochs (55).

```bash
python train_augmented.py
```

It saves to `model/cifar10_model_augmented.keras` (separate from the original,
so both are kept). Predict with it by passing the model path as a second
argument:

```bash
python predict.py sample_images/cat.png model/cifar10_model_augmented.keras
```

Results compared:

| Model | Test accuracy | Notes |
|-------|---------------|-------|
| `cifar10_model.keras` (15 epochs) | ~74% | overfits (memorizes training data) |
| `cifar10_model_augmented.keras` (55 epochs + augmentation) | ~75% | learns honestly, much better on the hard cat/dog classes |

The biggest win is on the easily-confused `cat` class: augmentation took its
confidence on a sample cat from ~4% up to ~69%.

---

## What to try next

- Change the number of `epochs` or `filters` and watch the accuracy change.
- Add another Conv2D block or `BatchNormalization` to push past 80%.
- Use a pre-trained model to handle full-size, real-world photos.
