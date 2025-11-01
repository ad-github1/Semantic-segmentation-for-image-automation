# Semantic Segmentation for Image Automation (ENet)

This project implements **semantic segmentation** using a lightweight **ENet architecture** trained on the **Cityscapes dataset** (20+ pixel-wise classes).  
Optimized for real-time edge deployment using TorchScript and OpenCV post-processing.

---

### Features
- Real-time inference (~25 FPS)
- IoU ≈ 0.78 on validation
- Background removal & object tagging automation
- Lightweight ENet backbone for mobile/edge devices

---

### Tech Stack
- Python 3.11+
- PyTorch
- OpenCV
- Streamlit
- NumPy, Pillow

---

### Setup
```bash
git clone https://github.com/ad-github1/Semantic-Segmentation-ENet.git
cd Semantic-Segmentation-ENet
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

