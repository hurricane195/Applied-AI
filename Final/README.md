# Multimodal Secure Access-Control System

This repository contains all scripts, datasets, and model files needed to replicate the multimodal access-control system developed for the ECGR 6119/8119 final project. The system integrates three AI modalities running on a Raspberry Pi 5 with a Hailo-8 accelerator:

1. YOLOv8n logo detection (Hailo-8 accelerated)
2. MediaPipe Hands gesture recognition
3. Whisper speech recognition

Only datasets small enough for GitHub are included. Required external datasets are linked below.

## System Overview

The objective of this system is to demonstrate secure, context-aware multimodal interaction. Logo detection serves as the authorization mechanism. When the UNCC logo is detected, gesture and speech commands become active and can control GPIO hardware. Real-time performance is achieved by compiling YOLOv8n to a Hailo Executable File (HEF) and running inference on the Hailo-8 accelerator.

---

## Repository Contents

### Datasets (Included)
- UNCC_Logo - Edited.zip
- UNCCAugmentedLogos2.zip

### Model Files
- Weights.zip (inculdes HEF file)

### Python Scripts
- YOLO_UNCC_LOGO_TRAINING_FOR_HAILO.py
- UNCC LOGO IMAGE AUGMENTATION.py
- RaspberryPi_YOLOv8n_for_Halo_MediaPipe_Whisper.py
- README.md (this file)

---

## Required External Datasets (Not Included)

1. Brand Logos and Icons Dataset  
   https://www.kaggle.com/datasets/programmer3/brand-logos-and-icons-dataset

2. University Student ID Card Dataset  
   https://www.kaggle.com/datasets/mirzamohibulhasan/university-student-id-card

3. HomeObjects-3K Dataset  
   https://docs.ultralytics.com/datasets/detect/homeobjects-3k/

These datasets supply negative samples for YOLO training.

---

## Hardware Requirements

- Raspberry Pi 5
- Hailo-8 M.2 AI Accelerator
- USB camera
- Microphone for Whisper speech recognition
- GPIO hardware (LEDs, relays, magnetic lock)
- Breadboard, jumper wires

---

## Software Requirements

Install dependencies using:

pip install -r requirements.txt

---

# Running the System

## 1. Extract the datasets

Unzip the following into your project directory:

- UNCC_Logo - Edited.zip
- UNCCAugmentedLogos2.zip
- Weights.zip

Ensure folder structures match the training script.

---

## 2. (Optional) Retrain YOLOv8n

Run:

python YOLO_UNCC_LOGO_TRAINING_FOR_HAILO.py

This can regenerate training outputs and ONNX weights for Hailo compilation.

---

## 3. (Optional) Regenerate Augmented Images

Run:

python "UNCC LOGO IMAGE AUGMENTATION.py"

This recreates the augmented logo dataset.

---

## 4. Run Raspberry Pi Inference (Hailo-8 + MediaPipe + Whisper)

Run:

python RaspberryPi_YOLOv8n_for_Halo_MediaPipe_Whisper.py --camera /dev/video1

This script performs:

- YOLOv8n logo detection on the Hailo-8 accelerator
- MediaPipe hand landmark and gesture processing
- Whisper speech recognition
- GPIO control gated by logo detection

---

## Reproducing the Full Negative Dataset

To reconstruct the full YOLOv8 training environment:

1. Download the three external datasets listed above.
2. Extract them into a directory such as negatives/.
3. Update any paths in the training script to match your directory layout.

These negative samples help reduce false positives during logo detection.

---

## Limitations

- Whisper accuracy may degrade in noisy environments.
- Gesture recognition depends on lighting and motion stability.
- Logo detection performance drops at extreme angles or with strong motion blur.
- Raspberry Pi compute limits restrict the size and complexity of additional models.

---

## Future Improvements

- Use beamforming microphones for improved Whisper robustness.
- Expand datasets to cover more lighting, backgrounds, and angles.
- Add additional authorization such as face recognition or ID text parsing.
- Deploy multiple Pi + Hailo devices for building-wide access control.

---

## Author

THunt 
ECGR 6119/8119 — Applied AI   
UNC Charlotte

---

## Contact

For questions or replication support, contact:  
mjhunt@uncc.edu
