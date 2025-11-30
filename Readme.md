# Face Recognition System with Deep Learning

A real-time face recognition system that uses deep learning models to detect and identify faces through webcam feed. The system uses OpenCV's DNN module with pre-trained models for face detection and face embedding extraction.

## Overview

This project implements a complete face recognition pipeline:
1. **Dataset Creation** - Extract face embeddings from images
2. **Model Training** - Train an SVM classifier on the embeddings
3. **Real-time Recognition** - Identify faces through webcam in real-time

## Features

- Real-time face detection and recognition via webcam
- Deep learning-based face detection using Caffe SSD model
- Face embedding extraction using OpenFace neural network
- SVM-based face classification
- Support for multiple people recognition

## Requirements

- Python 3.x
- OpenCV (opencv-contrib-python)
- NumPy
- imutils
- scikit-learn

Install dependencies:
```bash
pip install opencv-contrib-python numpy imutils scikit-learn
```

## Usage

### Quick Start

Simply run the batch file to execute the complete pipeline:
```bash
run_project.bat
```

### Step-by-Step Execution

1. **Create Dataset Embeddings**
   ```bash
   python buat_dataset.py --dataset dataset --embeddings output/embeddings.pickle --detector face_detector --embedding-model openface_nn4.small2.v1.t7
   ```

2. **Train the Recognizer**
   ```bash
   python pelatihan_dataset.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle
   ```

3. **Run Real-time Recognition**
   ```bash
   python pengenalan_realtime.py --detector face_detector --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle
   ```

Press `q` to quit the real-time recognition window.

## Adding New People

1. Create a new folder in `dataset/` with the person's name
2. Add multiple photos of the person to that folder
3. Re-run the complete pipeline or execute steps 1-2 from Step-by-Step Execution

## Models Used

- **Face Detector**: ResNet-10 SSD (300x300) - Caffe model
- **Face Embedder**: OpenFace nn4.small2.v1 - 128-dimensional face embeddings
- **Classifier**: Support Vector Machine (SVM) with linear kernel

## Notes

- The system uses webcam source 1 by default. Modify `src=1` in `pengenalan_realtime.py` if needed
- Minimum confidence threshold is set to 0.5 (50%) for face detection
- Ensure good lighting conditions for better recognition accuracy
