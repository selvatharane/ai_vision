import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision.models.segmentation import deeplabv3_resnet34
from PIL import Image, ImageDraw
from io import BytesIO
from streamlit_image_comparison import image_comparison
from skimage import measure
import zipfile
import time

# ================= Device =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= Enhanced Custom CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

/* Root variables for smooth theme transitions */
:root {
    --transition-speed: 0.4s;
    --shadow-light: 0 8px 32px rgba(0,0,0,0.08);
    --shadow-medium: 0 12px 48px rgba(0,0,0,0.12);
    --shadow-heavy: 0 20px 60px rgba(0,0,0,0.15);
}

/* Global smooth transitions */
* {
    transition: background-color var(--transition-speed) ease,
                color var(--transition-speed) ease,
                border-color var(--transition-speed) ease,
                box-shadow var(--transition-speed) ease,
                transform 0.3s ease;
}

/* Light mode */
@media (prefers-color-scheme: light) {
    .stApp { 
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 50%, #f1f3f5 100%);
        color: #212529;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .hero { 
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid rgba(230,57,70,0.1);
        box-shadow: var(--shadow-medium);
        border-radius: 24px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(230,57,70,0.05) 0%, transparent 70%);
        animation: pulse 8s ease-in-out infinite;
    }
    
    .hero h1 { 
        background: linear-gradient(135deg, #e63946 0%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }
    
    .hero p { 
        color: #6c757d;
        font-size: 1.2rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
        letter-spacing: 0.01em;
    }
    
    section[data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: 1px solid rgba(0,0,0,0.06);
        box-shadow: 4px 0 24px rgba(0,0,0,0.04);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] { 
        background: #f1f3f5;
        color: #495057;
        border-radius: 12px 12px 0 0;
        font-weight: 600;
        padding: 12px 24px;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #e9ecef;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] { 
        background: linear-gradient(135deg, #e63946 0%, #ff6b6b 100%) !important;
        color: white !important;
        box-shadow: 0 6px 20px rgba(230,57,70,0.35) !important;
        transform: translateY(-2px);
    }
    
    .upload-section {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: var(--shadow-light);
        border: 2px dashed #dee2e6;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #e63946;
        box-shadow: var(--shadow-medium);
    }
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
    .stApp { 
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f0f 100%);
        color: #e9ecef;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .hero { 
        background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%);
        border: 1px solid rgba(255,107,107,0.15);
        box-shadow: 0 12px 48px rgba(0,0,0,0.6);
        border-radius: 24px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,107,107,0.08) 0%, transparent 70%);
        animation: pulse 8s ease-in-out infinite;
    }
    
    .hero h1 { 
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8787 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }
    
    .hero p { 
        color: #adb5bd;
        font-size: 1.2rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
        letter-spacing: 0.01em;
    }
    
    section[data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #1a1a1a 0%, #0f0f0f 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
        box-shadow: 4px 0 24px rgba(0,0,0,0.5);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] { 
        background: #2a2a2a;
        color: #adb5bd;
        border-radius: 12px 12px 0 0;
        font-weight: 600;
        padding: 12px 24px;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #333333;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] { 
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8787 100%) !important;
        color: white !important;
        box-shadow: 0 6px 20px rgba(255,107,107,0.4) !important;
        transform: translateY(-2px);
    }
    
    .upload-section {
        background: #1a1a1a;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        border: 2px dashed #333333;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #ff6b6b;
        box-shadow: 0 12px 48px rgba(0,0,0,0.6);
    }
}

@keyframes pulse {
    0%, 100% { transform: scale(1) rotate(0deg); opacity: 1; }
    50% { transform: scale(1.1) rotate(5deg); opacity: 0.8; }
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

@keyframes shimmer {
    0% { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}

/* Enhanced Buttons */
.stButton > button {
    background: linear-gradient(135deg, #e63946 0%, #ff6b6b 100%);
    color: white !important;
    border-radius: 16px;
    font-weight: 700;
    font-size: 1.05rem;
    padding: 0.85rem 2rem;
    border: none;
    box-shadow: 0 6px 24px rgba(230,57,70,0.3);
    position: relative;
    overflow: hidden;
    letter-spacing: 0.02em;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s ease;
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button:hover,
.stButton > button:focus,
.stButton > button:active {
    transform: translateY(-3px) scale(1.02);
    background: linear-gradient(135deg, #ff6b6b 0%, #ff8787 100%);
    box-shadow: 0 12px 32px rgba(230,57,70,0.45);
    color: white !important;
}

/* Download Buttons */
.stDownloadButton > button {
    background: linear-gradient(135deg, #20c997 0%, #38d9a9 100%);
    color: white !important;
    border-radius: 14px;
    font-weight: 600;
    padding: 0.7rem 1.8rem;
    border: none;
    box-shadow: 0 4px 16px rgba(32,201,151,0.3);
}

.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(32,201,151,0.4);
    background: linear-gradient(135deg, #38d9a9 0%, #51cf66 100%);
    color: white !important;
}

/* Enhanced Images */
img {
    border-radius: 16px;
    box-shadow: var(--shadow-medium);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

img:hover {
    transform: scale(1.02);
    box-shadow: var(--shadow-heavy);
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #e63946, #ff6b6b, #ff8787);
    background-size: 200% 100%;
    animation: shimmer 2s infinite;
}

/* Radio buttons */
.stRadio > label {
    font-weight: 600;
    font-size: 1.05rem;
}

/* Sliders */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #e63946, #ff6b6b);
}

/* Sidebar header */
.css-1d391kg, [data-testid="stSidebarNav"] {
    padding-top: 2rem;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border-radius: 16px;
    padding: 1.5rem;
}

/* Checkboxes */
.stCheckbox {
    font-weight: 500;
}

/* Info messages */
.stAlert {
    border-radius: 12px;
    box-shadow: var(--shadow-light);
}

/* Spinner */
.stSpinner > div {
    border-top-color: #e63946 !important;
}
</style>
# ================= Hero Section =================
st.markdown("""
<div class="hero">
    <h1>AI Vision Extraction</h1>
    <p>Extract objects, edges, and apply artistic overlays with your images.</p>
</div>
""", unsafe_allow_html=True)

# ================= Model Loading =================
@st.cache_resource
def load_model():
    model = deeplabv3_resnet34(pretrained=False, num_classes=1)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    model.load_state_dict(torch.load("final_deeplab.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()
st.success("Model loaded successfully!")

# ================= Utility Functions =================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, (512, 512))
    image = image / 255.0
    image = image.transpose(2, 0, 1)
    return torch.tensor(image, dtype=torch.float32).unsqueeze(0)

def postprocess_mask(mask: torch.Tensor):
    mask = mask.squeeze().cpu().detach().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask

def overlay_mask(image: Image.Image, mask: np.ndarray):
    overlay = np.array(image.resize((mask.shape[1], mask.shape[0])))
    color_mask = np.zeros_like(overlay)
    color_mask[..., 0] = mask  # Red overlay
    overlayed = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)
    return Image.fromarray(overlayed)

def apply_edge(mask: np.ndarray):
    edges = cv2.Canny(mask, 100, 200)
    return edges

# ================= Upload Section =================
tab1, tab2 = st.tabs(["Single Image", "Batch Upload"])

with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Extract Objects"):
            with st.spinner("Processing..."):
                input_tensor = preprocess_image(image).to(device)
                with torch.no_grad():
                    output = model(input_tensor)['out']
                    mask = postprocess_mask(torch.sigmoid(output))
                overlayed = overlay_mask(image, mask)
                edges = apply_edge(mask)
                st.image(overlayed, caption="Segmentation Overlay", use_column_width=True)
                st.image(edges, caption="Edge Map", use_column_width=True)
                # Download buttons
                buffered = BytesIO()
                overlayed.save(buffered, format="PNG")
                st.download_button("Download Overlay", buffered.getvalue(), file_name="overlay.png", mime="image/png")
                buffered_edges = BytesIO()
                Image.fromarray(edges).save(buffered_edges, format="PNG")
                st.download_button("Download Edges", buffered_edges.getvalue(), file_name="edges.png", mime="image/png")

with tab2:
    uploaded_files = st.file_uploader("Upload multiple images", type=['png','jpg','jpeg'], accept_multiple_files=True)
    if uploaded_files:
        processed_images = []
        with st.spinner("Processing batch..."):
            for f in uploaded_files:
                img = Image.open(f)
                input_tensor = preprocess_image(img).to(device)
                with torch.no_grad():
                    output = model(input_tensor)['out']
                    mask = postprocess_mask(torch.sigmoid(output))
                overlayed = overlay_mask(img, mask)
                processed_images.append((f.name, overlayed))
        # Save all images to zip for download
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for name, img in processed_images:
                buf = BytesIO()
                img.save(buf, format="PNG")
                zip_file.writestr(f"{name.split('.')[0]}_overlay.png", buf.getvalue())
        st.download_button("Download All Overlays", zip_buffer.getvalue(), file_name="batch_overlays.zip", mime="application/zip")
        for _, img in processed_images:
            st.image(img, use_column_width=True)
            