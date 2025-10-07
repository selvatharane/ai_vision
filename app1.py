import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision.models.segmentation import deeplabv3_resnet34
from PIL import Image, ImageDraw
from io import BytesIO
from skimage import measure
import zipfile
import os

# ================= Device =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= Enhanced CSS =================
st.markdown(""" 
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

:root {
    --transition-speed: 0.4s;
    --shadow-light: 0 8px 32px rgba(0,0,0,0.08);
    --shadow-medium: 0 12px 48px rgba(0,0,0,0.12);
    --shadow-heavy: 0 20px 60px rgba(0,0,0,0.15);
}

* { transition: background-color var(--transition-speed) ease,
             color var(--transition-speed) ease,
             border-color var(--transition-speed) ease,
             box-shadow var(--transition-speed) ease,
             transform 0.3s ease; }

.stApp { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 50%, #f1f3f5 100%); color: #212529; font-family: 'Inter', sans-serif; }
.hero { background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%); border-radius: 24px; padding: 3rem 2rem; text-align: center; margin-bottom: 2rem; position: relative; overflow: hidden; box-shadow: var(--shadow-medium);}
.hero h1 { background: linear-gradient(135deg, #e63946 0%, #ff6b6b 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem;}
.hero p { color: #6c757d; font-size: 1.2rem;}
.stButton > button { background: linear-gradient(135deg, #e63946 0%, #ff6b6b 100%); color: white !important; border-radius: 16px; font-weight: 700; padding: 0.85rem 2rem; border: none; box-shadow: 0 6px 24px rgba(230,57,70,0.3); }
.stButton > button:hover { transform: translateY(-3px) scale(1.02); }
.stDownloadButton > button { background: linear-gradient(135deg, #20c997 0%, #38d9a9 100%); color: white !important; border-radius: 14px; padding: 0.7rem 1.8rem; border: none; box-shadow: 0 4px 16px rgba(32,201,151,0.3); }
.stDownloadButton > button:hover { transform: translateY(-2px); background: linear-gradient(135deg, #38d9a9 0%, #51cf66 100%); }
img { border-radius: 16px; box-shadow: var(--shadow-medium); transition: transform 0.3s ease, box-shadow 0.3s ease; }
img:hover { transform: scale(1.02); box-shadow: var(--shadow-heavy); }
</style>
""", unsafe_allow_html=True)

# ================= Load Model =================
@st.cache_resource
def load_model():
    num_classes = 2
    model_path = "final_deeplab.pth"
    model = deeplabv3_resnet34(weights=None, aux_loss=True)
    # Modify classifier for 2 classes
    old_cls = model.classifier
    model.classifier = nn.Sequential(
        old_cls[0], old_cls[1], old_cls[2], nn.Dropout(0.3),
        nn.Conv2d(256, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, num_classes, kernel_size=1)
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model

model = load_model()

# ================= Utilities =================
def preprocess(img: Image):
    img = img.convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()
    return img.to(device)

def tta_inference(img_tensor):
    """Simple TTA: original + horizontal flip"""
    with torch.no_grad():
        out1 = model(img_tensor)['out']
        out2 = model(torch.flip(img_tensor, [-1]))['out']
        out2 = torch.flip(out2, [-1])
        out = (out1 + out2)/2
        mask = torch.argmax(out.squeeze(), dim=0).cpu().numpy()
    return mask

def apply_artistic_effect(mask, img):
    color = np.array([255, 0, 0], dtype=np.uint8)
    img = np.array(img.resize((256,256)))
    mask_rgb = np.zeros_like(img)
    mask_rgb[mask==1] = color
    blended = cv2.addWeighted(img, 0.7, mask_rgb, 0.3, 0)
    return Image.fromarray(blended)

def extract_object(mask, img):
    img = np.array(img.resize((256,256)))
    result = np.zeros_like(img)
    result[mask==1] = img[mask==1]
    return Image.fromarray(result)

def get_edge_overlay(mask, img):
    edges = cv2.Canny((mask*255).astype(np.uint8), 100, 200)
    img = np.array(img.resize((256,256)))
    img[edges>0] = [0,255,0]
    return Image.fromarray(img)

# ================= Streamlit UI =================
st.title("🌟 AI Object Extraction App")
st.markdown('<div class="hero"><h1>AI Vision Extraction</h1><p>Upload an image and extract objects with AI</p></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Run Extraction"):
        with st.spinner("Processing..."):
            img_tensor = preprocess(img)
            mask = tta_inference(img_tensor)
            
            # Original + Artistic + Edge Overlay
            extracted = extract_object(mask, img)
            artistic = apply_artistic_effect(mask, img)
            edge_overlay = get_edge_overlay(mask, img)
            
            st.subheader("Results")
            st.image(extracted, caption="Extracted Object", use_column_width=True)
            st.image(artistic, caption="Artistic Overlay", use_column_width=True)
            st.image(edge_overlay, caption="Edge Overlay", use_column_width=True)
            
            # Download buttons
            buf1 = BytesIO()
            extracted.save(buf1, format="PNG")
            st.download_button("Download Extracted Object", data=buf1.getvalue(), file_name="extracted.png", mime="image/png")
            
            buf2 = BytesIO()
            artistic.save(buf2, format="PNG")
            st.download_button("Download Artistic Overlay", data=buf2.getvalue(), file_name="artistic.png", mime="image/png")
            
            buf3 = BytesIO()
            edge_overlay.save(buf3, format="PNG")
            st.download_button("Download Edge Overlay", data=buf3.getvalue(), file_name="edge_overlay.png", mime="image/png")
