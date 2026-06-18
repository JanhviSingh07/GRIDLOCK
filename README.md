# 🚦 Gridlock Shield
### AI-Powered Traffic Violation Detection System
**Flipkart Gridlock Hackathon 2.0 — Prototype Phase Submission**

---

## 🎯 Overview
Gridlock Shield is a computer vision-based prototype that automatically detects, 
classifies, and documents traffic violations from images and video footage.

## ✅ Features
- **Helmet Violation Detection** — YOLOv11n (mAP50: 0.744)
- **Triple Riding Detection** — YOLOv8n COCO + IoU logic
- **Number Plate Recognition** — YOLOv11n (mAP50: 0.993) + EasyOCR
- **Live Dashboard** — Streamlit with 5 pages
- **Enforcement Heatmap** — Bengaluru location-based priority
- **Impact Simulator** — Congestion reduction projections

## 🛠️ Tech Stack
- YOLOv11n / YOLOv8n (Ultralytics)
- EasyOCR, OpenCV, Streamlit
- Plotly, Folium, SQLite, Python 3.13

## 🚀 Run Locally
```bash
pip install ultralytics streamlit opencv-python pillow plotly folium streamlit-folium easyocr paddleocr paddlepaddle
streamlit run app.py
```

## 📊 Model Performance
| Model | mAP50 | Precision | Recall |
|-------|-------|-----------|--------|
| Helmet Detection | 0.744 | 0.824 | 0.669 |
| Number Plate | 0.993 | 0.990 | 0.976 |