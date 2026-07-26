# Project 4: SmartVision OpenCV Computer Vision Studio

A practical computer vision suite built with OpenCV and Streamlit featuring document boundary detection, perspective warping, HSV color segmentation, facial privacy blur, and digital image filter algorithms.

---

## 📌 Features
- **Document Edge Detection & Warp Scanner**: Canny edge extraction, contour approximation, and 4-point perspective transformation.
- **HSV Color Segmentation**: Interactive color tuning sliders (Hue, Saturation, Value) with automatic bounding box generation.
- **Facial Privacy Filter**: Automated Haar Cascade face localization and adaptive Gaussian anonymization blur.
- **Filter Studio**: Real-time Gaussian Blur, Sobel Gradients, Laplacian, Canny, and CLAHE Histogram Equalization.

---

## 📂 Folder Structure
```
04_OpenCV_Vision/
├── vision_processor.py   # OpenCV image processing engine
├── app.py                # Streamlit Web UI
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## 🚀 How to Run

```bash
streamlit run app.py
```
---
