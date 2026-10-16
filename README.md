# OpenEmotion

**Open-source facial emotion recognition using Deep Learning, PyTorch, OpenCV, and FER-2013.**

OpenEmotion is a deep-learning project that detects faces from images or a live webcam and predicts one of seven facial-expression classes.

> **Status:** 🚧 Active development

## Classroom backend API

The HTTP service now implements the backend's `/v1/analyze/frame` and
`/v1/analyze/video` contracts, plus `/v1/analyze/student-video` for grouped
student timelines. It includes authentication, label mapping, signed video
downloads and durable idempotency. See the
[video emotion analysis README](docs/video-emotion-analysis/README.md) and
[backend integration instructions](docs/backend-integration.md).

## Demo

OpenEmotion can process a live camera feed using OpenCV:

```text
Webcam
   ↓
OpenCV
   ↓
Face Detection
   ↓
Face Preprocessing
   ↓
EfficientNet-B0
   ↓
Emotion + Confidence
```

## Supported Emotions

| Emotion | Class    |
| ------- | -------- |
| 😠      | Angry    |
| 🤢      | Disgust  |
| 😨      | Fear     |
| 😀      | Happy    |
| 😐      | Neutral  |
| 😢      | Sad      |
| 😲      | Surprise |

## Features

* 🧠 PyTorch deep-learning model
* 👤 Face detection using OpenCV
* 🎭 7-class facial-expression classification
* 📷 Real-time webcam detection
* 🖼️ Image-based inference
* ⚡ GPU support with CUDA
* 💻 CPU support
* 📊 Classification report
* 📈 Confusion matrix
* 📓 Kaggle/Jupyter training notebook
* 🔧 Modular Python source code
* 🧪 Basic automated tests

---

# Tech Stack

* **Python**
* **PyTorch**
* **TorchVision**
* **OpenCV**
* **EfficientNet-B0**
* **FER-2013**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**

---

# Project Structure

```text
OpenEmotion/
│
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── notebooks/
│   └── emotion_detection.ipynb
│
├── src/
│   └── openemotion/
│       ├── __init__.py
│       ├── config.py
│       ├── model.py
│       ├── preprocessing.py
│       ├── inference.py
│       └── face_detector.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── webcam.py
│
├── tests/
│   └── test_inference.py
│
├── models/
│   └── .gitkeep
│
└── docs/
    └── architecture.md
```

---

# Requirements

Recommended:

* Python **3.10+**
* Git
* 8 GB+ RAM
* NVIDIA GPU + CUDA for faster training — optional
* Webcam for real-time detection — optional

OpenEmotion can run on CPU, but training will be considerably slower.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/sharmaB01/OpenEmotion.git
```

Enter the project directory:

```bash
cd OpenEmotion
```

---

## 2. Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\activate
```

---

## 3. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset

OpenEmotion uses the **FER-2013** dataset.

The dataset is **not included in this GitHub repository**.

https://www.kaggle.com/datasets/msambare/fer2013
Download the FER-2013 dataset from a legitimate source and arrange it in the following structure:

```text
fer2013/
│
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

Each directory should contain the corresponding facial-expression images.

For example:

```text
fer2013/train/happy/
├── image1.jpg
├── image2.jpg
├── image3.jpg
└── ...
```

---

# Training

OpenEmotion currently uses **EfficientNet-B0** as its baseline model.

To train:

```bash
python scripts/train.py \
    --data /path/to/fer2013 \
    --epochs 10
```

Example:

```bash
python scripts/train.py \
    --data ~/datasets/fer2013 \
    --epochs 10
```

You can change the batch size:

```bash
python scripts/train.py \
    --data ~/datasets/fer2013 \
    --epochs 10 \
    --batch-size 64
```

After training, the model is saved to:

```text
models/openemotion_efficientnet_b0.pth
```

---

# Training on Kaggle

If you don't have a local GPU, you can train OpenEmotion using **Kaggle**.

The repository contains the training notebook:

```text
notebooks/emotion_detection.ipynb
```

Upload the notebook to Kaggle and attach a FER-2013 dataset.

Kaggle GPU can then be used for training.

After training, download the generated model:

```text
openemotion_efficientnet_b0.pth
```

Place it in:

```text
models/
```

Resulting structure:

```text
OpenEmotion/
└── models/
    └── openemotion_efficientnet_b0.pth
```

You can then use the trained model with the local webcam application.

---

# Evaluate the Model

After training:

```bash
python scripts/evaluate.py \
    --data /path/to/fer2013 \
    --model models/openemotion_efficientnet_b0.pth
```

Example:

```bash
python scripts/evaluate.py \
    --data ~/datasets/fer2013 \
    --model models/openemotion_efficientnet_b0.pth
```

The evaluation provides:

* Precision
* Recall
* F1-score
* Classification report
* Confusion matrix

---

# Real-Time Webcam Detection

Once you have a trained model, start the webcam application:

```bash
python scripts/webcam.py \
    --model models/openemotion_efficientnet_b0.pth
```

OpenEmotion will:

```text
Webcam
   ↓
OpenCV Frame
   ↓
Face Detection
   ↓
Face Crop
   ↓
Image Preprocessing
   ↓
EfficientNet-B0
   ↓
Emotion Prediction
   ↓
Confidence Score
```

The webcam window will display something similar to:

```text
┌─────────────────────────────┐
│                             │
│      ┌───────────────┐      │
│      │               │      │
│      │     FACE      │      │
│      │               │      │
│      └───────────────┘      │
│                             │
│      Happy 87.4%            │
│                             │
└─────────────────────────────┘
```

Press **`Q`** to close the application.

---

# Multiple Cameras

The default camera is:

```text
0
```

To use another camera:

```bash
python scripts/webcam.py \
    --model models/openemotion_efficientnet_b0.pth \
    --camera 1
```

---

# GPU Support

OpenEmotion automatically checks whether CUDA is available.

You can check your PyTorch installation:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If you get:

```text
True
```

PyTorch can use your NVIDIA GPU.

If you get:

```text
False
```

the project will run on CPU.

---

# Jupyter Notebook

The complete experimentation notebook is available at:

```text
notebooks/emotion_detection.ipynb
```

Start Jupyter:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

The notebook can be used for:

* Dataset exploration
* Data visualization
* Model training
* Model evaluation
* Classification reports
* Confusion matrices
* Experimentation

For normal usage, the Python scripts are recommended.

---

# Python API

OpenEmotion can also be used directly from Python.

Example:

```python
import torch

from openemotion.model import load_model
from openemotion.inference import predict_emotion

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = load_model(
    "models/openemotion_efficientnet_b0.pth",
    device,
    7
)

emotion, confidence = predict_emotion(
    model,
    face_image,
    device
)

print("Emotion:", emotion)
print("Confidence:", confidence)
```

---

# Run Tests

Install pytest if required:

```bash
pip install pytest
```

Run:

```bash
pytest
```

---

# Model Details

| Component      | Value               |
| -------------- | ------------------- |
| Model          | EfficientNet-B0     |
| Framework      | PyTorch             |
| Input          | 224 × 224 RGB       |
| Output         | 7 classes           |
| Dataset        | FER-2013            |
| Face Detection | OpenCV Haar Cascade |
| GPU            | CUDA supported      |
| CPU            | Supported           |

---

# Limitations

Facial-expression recognition is an estimation problem and is not perfectly reliable.

Performance can be affected by:

* Lighting
* Camera quality
* Face angle
* Occlusion
* Multiple faces
* Image quality
* Facial appearance
* Dataset bias
* Differences between training and real-world images

The predicted class should be considered a **model prediction**, not a definitive measurement of a person's actual emotional state.

---

# Roadmap

* [x] FER-2013 dataset support
* [x] PyTorch training
* [x] EfficientNet-B0 baseline
* [x] Model evaluation
* [x] Classification report
* [x] Confusion matrix
* [x] OpenCV face detection
* [x] Real-time webcam prototype
* [ ] Improve model accuracy
* [ ] Better face detection
* [ ] Multi-face tracking
* [ ] FPS optimization
* [ ] Image inference CLI
* [x] Video inference
* [ ] ONNX export
* [x] FastAPI inference API
* [ ] React web application
* [ ] Docker support
* [ ] CI/CD
* [ ] Model benchmarking
* [ ] Improved automated tests

---

# Contributing

Contributions are welcome!

## Fork the repository

```bash
git clone https://github.com/sharmaB01/OpenEmotion.git
cd OpenEmotion
```

Create a feature branch:

```bash
git checkout -b feature/my-feature
```

Make your changes and run the tests:

```bash
pytest
```

Commit your changes:

```bash
git add .
git commit -m "Add my feature"
```

Push the branch:

```bash
git push origin feature/my-feature
```

Then open a Pull Request.

---

# License

OpenEmotion is released under the **MIT License**.

See [LICENSE](LICENSE) for details.

---

# Acknowledgements

OpenEmotion is built using:

* [PyTorch](https://pytorch.org/)
* [TorchVision](https://pytorch.org/vision/)
* [OpenCV](https://opencv.org/)
* [Scikit-learn](https://scikit-learn.org/)
* FER-2013

---

# Disclaimer

OpenEmotion is an open-source research and educational project.

Facial-expression classification is an estimation task and should not be interpreted as a reliable measurement of a person's internal emotional state.

Do not use the model as the sole basis for decisions involving employment, education, healthcare, law enforcement, financial services, or other high-impact decisions.

---

# Quick Start

For experienced users:

```bash
git clone https://github.com/sharmaB01/OpenEmotion.git

cd OpenEmotion

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

Prepare FER-2013:

```text
fer2013/
├── train/
└── test/
```

Train:

```bash
python scripts/train.py \
    --data /path/to/fer2013 \
    --epochs 10
```

Evaluate:

```bash
python scripts/evaluate.py \
    --data /path/to/fer2013 \
    --model models/openemotion_efficientnet_b0.pth
```

Run webcam:

```bash
python scripts/webcam.py \
    --model models/openemotion_efficientnet_b0.pth
```

Press **Q** to exit.

---

## OpenEmotion

**PyTorch + OpenCV + EfficientNet-B0 + FER-2013**

⭐ If you find this project useful, consider giving the repository a star.
