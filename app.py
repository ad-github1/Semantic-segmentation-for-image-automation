import streamlit as st
import cv2
import numpy as np
from PIL import Image
from enet_model import ENetONNX

st.title("ENet Semantic Segmentation Demo")

# Load model
model = ENetONNX("models/enet_cityscapes.onnx")

uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with st.spinner("Running segmentation..."):
        seg, time_ms = model.infer(img_bgr)
        seg_rgb = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)

    st.image([image, seg_rgb], caption=["Original", f"Segmented ({time_ms:.1f} ms)"])
