# Industrial-Clamp-Detection-and-Spatial-Measurement-System-using-YOLO11
An industrial computer vision system for automatic clamp and wire detection using YOLO11, followed by spatial distance measurement, clamp integrity verification, and real-time quality inspection from production line videos.


# 🔩 Industrial Clamp Inspection using YOLO11

## 📌 Overview

This project presents a complete **Industrial Computer Vision** pipeline designed for automated quality inspection in a manufacturing environment.

The system analyses production-line videos captured by a high-resolution industrial camera and performs **object detection, spatial analysis, and geometric measurements** to verify the correct assembly of metal clamps and surrounding wires.

Rather than performing object detection alone, the project integrates **deep learning** with **engineering measurement algorithms** to deliver a reliable industrial quality inspection solution.

---

# 🏭 Industrial Scenario

The production machine contains a rotating circular component that holds multiple **metal clamps** together with **seven surrounding wires**.

Each clamp is divided into three independent components:

- **R1** – Clamp Head
- **R2** – Clamp Body
- **H** – Clamp Height

In addition, the system detects:

- **Wire**

Therefore, the trained YOLO11 model detects the following **four object classes**:

| Class | Description |
|--------|-------------|
| R1 | Clamp Head |
| R2 | Clamp Body |
| H | Clamp Height |
| Wire | Industrial Wire |

---

# ⚙️ Project Workflow

The complete inspection pipeline consists of the following stages.

## 1️⃣ Video Acquisition

Production videos are captured using a high-resolution industrial camera installed in front of the manufacturing machine.

---

## 2️⃣ Frame Extraction

Thousands of frames are extracted from the production videos to build the training dataset.

---

## 3️⃣ Dataset Annotation

All extracted images are manually annotated using **Roboflow**.

Bounding boxes are created for:

- R1
- R2
- H
- Wire

---

## 4️⃣ Dataset Preparation

The dataset is divided into:

- Training Set
- Validation Set
- Test Set

All images are resized to:

**1280 × 1280 pixels**

---

## 5️⃣ Model Training

The object detector is trained using **YOLO11**.

The model is trained to detect:

- Clamp Head (R1)
- Clamp Body (R2)
- Clamp Height (H)
- Wires

---

## 6️⃣ Clamp Integrity Verification

A clamp is considered **valid** only when all three components are detected simultaneously.

```
R1
+
R2
+
H
```

If one or more components are missing, the clamp is considered **incomplete**, and the inspection fails for that clamp.

---

## 7️⃣ Wire Detection

The system verifies that **all seven wires** are successfully detected.

Missing wire detections indicate a failed inspection.

---

## 8️⃣ Spatial Measurement

Once all objects have been detected successfully, the system performs several geometric measurements.

The calculated measurements include:

- Distance between each wire and its nearest **R2** component.
- Shortest wire distance.
- Longest wire distance.
- Height of the **H** bounding box.

---

## 9️⃣ Pixel-to-Centimetre Conversion

All measurements are initially calculated in **pixels**.

A calibration process converts pixel measurements into **real-world centimetres**.

---

## 🔟 Video Inference

The complete inspection pipeline is evaluated using full production videos to verify:

- Detection accuracy
- Clamp verification
- Distance measurements
- Industrial robustness
- Real-time performance

---

# 📊 Model Performance

| Metric | Score |
|---------|-------:|
| Precision | **0.8547** |
| Recall | **1.0000** |
| mAP@50 | **0.9825** |
| mAP@50-95 | **0.7762** |

---

# 🚧 Challenges

## 📉 Limited Dataset

During the early stages of development, the dataset contained a limited number of annotated images.

This resulted in unstable **wire detection** performance, while clamp detection remained consistently accurate.

The issue was resolved by improving the dataset quality and increasing the number of annotated training images.

---

## 🔄 Detection Ordering

The spatial measurement stage required a strict processing order.

Distance calculations were only valid when **wires were processed before clamps**.

A dedicated post-processing algorithm was implemented to correctly associate each wire with its nearest clamp before calculating geometric measurements.

---

# ✅ Industrial Validation

The final system was successfully tested on complete production videos.

The results demonstrated reliable:

- Clamp detection
- Wire detection
- Clamp integrity verification
- Spatial distance measurement
- Pixel-to-centimetre conversion

making the system suitable for **real-world industrial quality inspection applications**.

---

# 🛠️ Technologies Used

- Python
- YOLO11 (Ultralytics)
- OpenCV
- Roboflow
- NumPy
- Google Colab
- Computer Vision
- Deep Learning

---

# 🎯 Applications

This project can be applied to:

- Industrial Quality Inspection
- Smart Manufacturing
- Automated Production Lines
- Defect Detection
- Industrial Computer Vision
- Factory Automation
- Industry 4.0 Systems
- AI-based Visual Inspection

---


---

# 🧩 Software Architecture

The project was designed using a modular programming approach to improve code readability, maintainability, and scalability.

Instead of implementing the entire pipeline in a single script, the project was organized into reusable functions responsible for different processing stages.

The implementation includes dedicated functions for:

- Model loading and initialization.
- Video acquisition and frame processing.
- Object detection using YOLO11.
- Detection filtering and object classification.
- Clamp integrity verification.
- Reference clamp selection.
- Spatial distance calculations.
- Pixel-to-centimetre conversion.
- Wire measurement analysis.
- Bounding box visualization.
- Detection labeling.
- Video rendering and output generation.

This modular design makes the project easier to maintain, debug, extend, and integrate into industrial inspection systems.

---


# ⭐ Key Features

- Custom YOLO11 object detection model.
- Four-class industrial object detection.
- Automatic clamp integrity verification.
- Seven-wire detection and validation.
- Intelligent post-processing algorithms.
- Spatial distance measurement.
- Pixel-to-centimetre calibration.
- Modular software architecture.
- Video-based industrial inspection.
- Automatic visualization of measurements.
- Real-time processing pipeline.
- Production-ready workflow.

---
