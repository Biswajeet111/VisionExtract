# VisionExtract: Subject Isolation using Image Segmentation

## 📌 Project Overview

VisionExtract is a computer vision project aimed at building a deep learning-based system that automatically extracts the main subject from an image. 

For any given input image, the system will output a new image where:
- The main subject remains unchanged
- All background pixels are converted to black

This is achieved using semantic segmentation techniques trained on the COCO 2017 dataset.

---

## 🎯 Problem Statement

The goal of this project is to develop an end-to-end pipeline capable of:

1. Processing annotated image datasets
2. Generating binary segmentation masks
3. Training a segmentation model
4. Performing inference on unseen images
5. Producing subject-isolated outputs

---

## 📂 Dataset

We are using the **COCO 2017 Dataset** for training and experimentation.

### Dataset Components Used:
- `train2017` (Image files)
- `annotations/instances_train2017.json` (Segmentation annotations)

Dataset structure inside project:


data/
├── train2017/
└── annotations/
├── instances_train2017.json


---

## 🏗️ Project Structure


VisionExtract/
│
├── data/ # Dataset storage (not pushed to GitHub)
├── src/ # Core source code
│ ├── dataset.py
│ ├── preprocessing.py
│ ├── utils.py
│ ├── train.py
│ └── inference.py
│
├── notebooks/ # Experimental development
├── outputs/ # Generated results
├── checkpoints/ # Model weights
├── venv/ # Virtual environment (ignored)
├── requirements.txt
└── README.md


---

## 🛠️ Week 1 Milestone: Project Initialization & Dataset Setup

### ✅ Completed Tasks:

- Project structure created
- Virtual environment configured (Python 3.10)
- Required dependencies installed
- COCO 2017 dataset downloaded
- Annotation files extracted and validated
- Git version control setup 
- Feature branch workflow initialized

---

## 🔍 Current Focus (Week 1)

- Dataset exploration using COCO API
- Visualization of images and segmentation masks
- Converting multi-class masks into binary masks
- Preparing clean data preprocessing pipeline

---

## 🚀 Next Steps

- Implement dataset loader
- Generate binary subject masks
- Prepare model-ready input pipeline
- Begin baseline segmentation model implementation

---

## 📊 Evaluation Metrics (Planned)

- Intersection over Union (IoU)
- Dice Coefficient
- Pixel-wise Accuracy

---

## 👨‍💻 Author

Internship Project – VisionExtract  
Subject Isolation using Deep Learning Segmentation
