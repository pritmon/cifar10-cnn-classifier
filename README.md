<div align="center">

# 🖼️ CIFAR-10 CNN Image Classifier

**Teaching a machine to see — a clean, fully-documented Convolutional Neural Network that sorts tiny photos into 10 categories.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Sequential%20API-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-75.4%25-success)](#-results)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## 🎯 Mission

Most deep-learning examples hand you a black box. This project does the
opposite: it is a **small, honest, end-to-end image classifier where every
design decision is explained** — readable code, a measurable baseline, and a
clear path from a naïve model to a stronger one. The goal is not just a
classifier, but a transparent reference you can read, run, and extend.

It classifies 32×32 colour images into **10 classes**:

`airplane` · `automobile` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`

---

## ✨ Highlights

- 🧱 **3-block CNN** built with the clean Keras `Sequential` API
- 📝 **Every block commented** — written to be read, not just run
- 🔁 **Two trainers**: a baseline (`train.py`) and an augmented, higher-accuracy version (`train_augmented.py`)
- 🎛️ **Data augmentation** (flip / rotate / zoom) that measurably improves the hardest classes
- 🔮 **One-command prediction** on any image, with top-3 guesses and confidence
- 📦 **Trained models included** — predict immediately, no training required

---

## 🚀 Quickstart

```bash
# 1. Clone and enter
git clone https://github.com/pritmon/cifar10-cnn-classifier.git
cd cifar10-cnn-classifier

# 2. Set up an isolated environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Predict right away (a trained model ships with the repo)
python predict.py sample_images/ship.png
```

Example output:

```
Predicted class: ship
Confidence: 97.41%

Top 3 guesses:
  1. ship         97.41%
  2. automobile    2.59%
  3. airplane      0.00%
```

Want to train from scratch instead? → `python train.py`

---

## 🧠 How it works

The model reads an image as a stack of numbers and passes it through three
"detector" blocks. Each block finds richer patterns while shrinking the image,
so deeper layers see the *big picture* rather than single pixels.

```
Input 32×32×3
  │
  ├─ Conv2D(32)  → MaxPool   ·  finds edges & colours      ·  32×32 → 16×16
  ├─ Conv2D(64)  → MaxPool   ·  finds shapes               ·  16×16 → 8×8
  ├─ Conv2D(128) → MaxPool   ·  finds parts (ears, wheels) ·  8×8 → 4×4
  │
  ├─ Flatten → Dense(128)    ·  the learned "fingerprint" of the image
  ├─ Dropout(0.3)            ·  prevents memorising (regularisation)
  └─ Dense(10, softmax)      ·  probability for each of the 10 classes
```

Pixels are normalised to the `0–1` range before training for stable learning.

---

## 📊 Results

| Model | Training | Test accuracy | Notes |
|-------|----------|:---:|-------|
| `cifar10_model.keras` | 15 epochs | **74.4%** | Strong baseline, but overfits the training data |
| `cifar10_model_augmented.keras` | 55 epochs + augmentation | **75.4%** | Learns honestly; far stronger on confusable classes |

**The augmentation win, made concrete** — on a sample `cat` image, model
confidence in the correct class climbed dramatically as augmentation and
training were added:

```
cat confidence:   3.9%  →  34.1%  →  68.7%   (finally the #1 prediction)
```

> CIFAR-10's `cat` vs `dog` is famously the hardest pair to separate at 32×32.
> Closing that gap is exactly where data augmentation pays off most.

---

## 🛠️ The augmented model

`train_augmented.py` adds three augmentation layers at the front of the network:

```python
layers.RandomFlip("horizontal")   # mirror left–right
layers.RandomRotation(0.1)        # tilt slightly
layers.RandomZoom(0.1)            # zoom in / out
```

These randomly perturb every image *during training only*, so the model never
sees the same picture twice — forcing it to learn the underlying object instead
of memorising specific photos.

```bash
python train_augmented.py
# then predict with the augmented model:
python predict.py sample_images/cat.png model/cifar10_model_augmented.keras
```

---

## 📁 Project structure

```
cifar10-cnn-classifier/
├── train.py                 # baseline trainer (build → train → save)
├── train_augmented.py       # augmented trainer (higher accuracy)
├── predict.py               # load a model, classify any image
├── make_sample_images.py    # save a few real CIFAR-10 images to test on
├── model/                   # trained models (.keras)
├── sample_images/           # ready-to-use test images
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🗺️ Roadmap

- [ ] Add `BatchNormalization` and a 4th conv block to push past 80%
- [ ] Fine-tune a pre-trained backbone to handle full-resolution, real-world photos
- [ ] Ship a drag-and-drop web demo

---

## 📄 License

Released under the [MIT License](LICENSE) — free to use, learn from, and build on.

<div align="center">
<sub>A hands-on study of convolutional neural networks — built, measured, and improved. 🐱</sub>
</div>
