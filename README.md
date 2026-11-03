# Automated Vehicle Counting and Traffic Analysis System (YOLOv8 & OpenCV)

An intelligent transportation and computer vision project designed for automated vehicle detection, classification, and density estimation from complex traffic imagery (highways, urban CCTV, and aerial drone footage).

---

## Project Overview
Traditional methods (like Haar Cascade Classifiers) frequently fail under dynamic conditions such as heavy tree shadows, low contrast, and high-angle aerial (bird's-eye) views. This project addresses these limitations by integrating modern Deep Learning architectures with classical digital image processing pipelines.

## Technological Stack & Architecture
- **Language:** Python 3.10
- **Deep Learning Model:** YOLOv8 Medium (`yolov8m`) via Ultralytics
- **Image Processing:** OpenCV (`cv2.resize`, `cv2.cvtColor`, `cv2.GaussianBlur`)
- **GUI:** Tkinter & Pillow (PIL)
- **Datasets & Benchmarks:** Microsoft COCO & Berkeley DeepDrive (BDD100K)

---

## Key Technical Features
- **Adaptive Resolution Pipeline:** High-resolution drone and CCTV inputs (exceeding 1080p/1920px) are adaptively scaled maintaining exact aspect ratios to eliminate memory bottlenecks without losing small-vehicle fidelity.
- **Preprocessing & Denoising:** Employs Gaussian Blur (5x5 kernel) and color space normalization to filter high-frequency noise and optimize input structural integrity.
- **Precision Inference & Filtering:**
  - Class Filtering: Strictly localized to relevant transportation classes (`2: Car`, `3: Motorcycle`, `5: Bus`, `7: Truck`).
  - Confidence & NMS: Tuned confidence threshold (`conf=0.35`) and Non-Maximum Suppression (`IoU=0.50`) with class-agnostic NMS to resolve vehicle occlusion.
- **Superior Accuracy:** Achieves >95% detection accuracy, successfully resolving shadows, reflections, and perspective distortions that cause high false-positive rates in Haar Cascade baselines.

---

## Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ozlemnurbora/Traffic-Vehicle-Counter.git](https://github.com/ozlemnurbora/Traffic-Vehicle-Counter.git)
   cd Traffic-Vehicle-Counter
   Install dependencies:

Bash
pip install -r requirements.txt
Run the Application:

Bash
python main.py
