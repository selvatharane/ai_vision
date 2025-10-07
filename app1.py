import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import segmentation_models_pytorch as smp
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
""", unsafe_allow_html=True)

# ================= Load Model =================
@st.cache_resource
def load_model(model_path="best_deeplab.pth", num_classes=2):
    import os
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None

    # Create DeepLabV3 model
    model = smp.DeepLabV3(
        encoder_name="resnet34",
        encoder_weights=None,  # or "imagenet" for pretrained encoder
        in_channels=3,
        classes=num_classes,
        aux_params=None      # Ensures no auxiliary outputs
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model

# ================= Artistic Effects Functions =================
def apply_sketch_effect(img, intensity=1.0):
    """Apply pencil sketch effect"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inv_gray = 255 - gray
    blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256.0)
    sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    
    # Blend with original based on intensity
    result = cv2.addWeighted(img, 1 - intensity, sketch_rgb, intensity, 0)
    return result.astype(np.uint8)

def apply_cartoon_effect(img, intensity=1.0):
    """Apply cartoon effect using bilateral filter and edge detection"""
    # Reduce colors
    num_down = 2
    num_bilateral = 7
    
    img_color = img.copy()
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
    
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
    
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)
    
    # Resize to match original
    img_color = cv2.resize(img_color, (img.shape[1], img.shape[0]))
    
    # Edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 9, 2)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # Combine
    cartoon = cv2.bitwise_and(img_color, edges)
    
    # Blend with original based on intensity
    result = cv2.addWeighted(img, 1 - intensity, cartoon, intensity, 0)
    return result.astype(np.uint8)

def apply_oil_paint_effect(img, intensity=1.0):
    """Apply oil painting effect using bilateral filter"""
    # Multiple passes of bilateral filter for oil painting effect
    result = img.copy()
    
    # Number of iterations based on intensity
    iterations = int(3 + 2 * intensity)
    
    for i in range(iterations):
        result = cv2.bilateralFilter(result, 9, 75, 75)
    
    # Add slight median blur for smoother look
    result = cv2.medianBlur(result, 5)
    
    # Enhance saturation for oil paint look
    hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # Blend with original based on intensity
    result = cv2.addWeighted(img, 1 - intensity, result, intensity, 0)
    return result.astype(np.uint8)

def apply_watercolor_effect(img, intensity=1.0):
    """Apply watercolor effect"""
    # Stylization gives a watercolor-like effect
    stylized = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
    
    # Blend with original based on intensity
    result = cv2.addWeighted(img, 1 - intensity, stylized, intensity, 0)
    return result.astype(np.uint8)

def apply_pencil_color_effect(img, intensity=1.0):
    """Apply colored pencil effect"""
    _, pencil_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    
    # Blend with original based on intensity
    result = cv2.addWeighted(img, 1 - intensity, pencil_color, intensity, 0)
    return result.astype(np.uint8)

def apply_artistic_effect(img, effect_type, intensity=1.0):
    """Apply selected artistic effect to image"""
    if effect_type == "None":
        return img
    elif effect_type == "Sketch":
        return apply_sketch_effect(img, intensity)
    elif effect_type == "Cartoon":
        return apply_cartoon_effect(img, intensity)
    elif effect_type == "Oil Painting":
        return apply_oil_paint_effect(img, intensity)
    elif effect_type == "Watercolor":
        return apply_watercolor_effect(img, intensity)
    elif effect_type == "Colored Pencil":
        return apply_pencil_color_effect(img, intensity)
    return img

# ================= NEW: Edge Overlay Function =================
def create_edge_overlay(image, mask, edge_color="#00FF00", edge_thickness=2):
    """Create edge overlay visualization on the original image"""
    # Convert color from hex to RGB tuple
    edge_color_rgb = tuple(int(edge_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    
    # Convert image to PIL if it's numpy array
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Resize mask to match image size if needed
    if isinstance(mask, np.ndarray):
        mask_resized = Image.fromarray((mask * 255).astype(np.uint8))
    else:
        mask_resized = mask
    
    if mask_resized.size != pil_image.size:
        mask_resized = mask_resized.resize(pil_image.size, Image.NEAREST)
    
    # Find contours using skimage
    mask_array = np.array(mask_resized)
    contours = measure.find_contours(mask_array, 128)
    
    # Create overlay
    overlay_edges = pil_image.copy()
    draw = ImageDraw.Draw(overlay_edges)
    
    for contour in contours:
        # Scale contour to match original image (already matched, but kept for compatibility)
        contour_scaled = contour * (pil_image.size[0] / mask_resized.width)
        # Swap coordinates (y,x → x,y)
        contour_points = [tuple(p[::-1]) for p in contour_scaled]
        
        if len(contour_points) > 1:
            draw.line(contour_points, fill=edge_color_rgb, width=edge_thickness)
    
    return np.array(overlay_edges)

# ================= Preprocess =================
def preprocess_image(img, long_side=256):
    img = np.array(img)
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    new_w, new_h = int(w*scale), int(h*scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    img_tensor = torch.tensor(img_resized/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor.to(device), (h, w), img

# ================= TTA Prediction =================
def tta_predict(img_tensor):
    """
    Test-Time Augmentation (TTA) prediction for DeepLabV3.
    Applies flips and rotations to improve segmentation quality.
    """
    aug_list = [
        lambda x: x,                         # original
        lambda x: torch.flip(x, [3]),        # horizontal flip
        lambda x: torch.flip(x, [2]),        # vertical flip
        lambda x: torch.rot90(x, k=1, dims=[2,3]),  # rotate 90
        lambda x: torch.rot90(x, k=3, dims=[2,3])   # rotate 270
    ]

    outputs = []
    with torch.no_grad():
        for i, f in enumerate(aug_list):
            t = f(img_tensor)
            out = model(t)  # DeepLabV3 output is already a tensor
            
            # Reverse augmentation
            if i == 1:       # horizontal flip
                out = torch.flip(out, [3])
            elif i == 2:     # vertical flip
                out = torch.flip(out, [2])
            elif i == 3:     # rotate 90
                out = torch.rot90(out, k=3, dims=[2,3])
            elif i == 4:     # rotate 270
                out = torch.rot90(out, k=1, dims=[2,3])
            
            outputs.append(out)

    # Average predictions
    return torch.mean(torch.stack(outputs), dim=0)


# ================= Post-process mask =================
def postprocess_mask(mask_tensor, orig_size, blur_strength=5):
    mask = torch.argmax(mask_tensor, dim=1).squeeze().cpu().numpy()
    mask = cv2.resize(mask.astype(np.uint8), (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = mask.astype(np.uint8) * 255
    blurred = cv2.GaussianBlur(mask, (blur_strength, blur_strength), 0)
    _, smooth_mask = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
    return (smooth_mask // 255).astype(np.uint8)

# ================= Extract Object =================
def extract_object(img, mask, bg_color=(0,0,0), transparent=False, custom_bg=None, gradient=None, bg_blur=False, blur_amount=21):
    mask = mask.astype(np.float32)
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    mask = np.expand_dims(mask, axis=-1)
    mask = mask / (mask.max() + 1e-8)
    if transparent:
        rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA).astype(np.float32)
        rgba[..., 3] = (mask.squeeze() * 255).astype(np.uint8)
        return rgba.astype(np.uint8)
    else:
        if bg_blur:
            # Create blurred background from original image
            bg_resized = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0).astype(np.float32)
        elif gradient is not None:
            h, w = img.shape[:2]
            col1, col2 = gradient
            bg_resized = np.zeros((h, w, 3), dtype=np.float32)
            for i in range(h):
                alpha = i / max(h-1, 1)
                row_color = (1 - alpha) * col1.astype(np.float32) + alpha * col2.astype(np.float32)
                bg_resized[i, :, :] = row_color
        elif custom_bg is not None:
            bg_resized = cv2.resize(np.array(custom_bg), (img.shape[1], img.shape[0])).astype(np.float32)
        else:
            bg_resized = np.full_like(img, bg_color, dtype=np.float32)
        result = img.astype(np.float32) * mask + bg_resized.astype(np.float32) * (1 - mask)
        return result.astype(np.uint8)

# ================= Hero Section =================
st.markdown("""
<div class='hero'>
    <h1>✨ AI Object Segmentation</h1>
    <p>Transform your images with precision AI-powered object extraction</p>
</div>
""", unsafe_allow_html=True)

# === Centered Banner Image ===
st.markdown(
    """
    <div style='text-align: center; margin-top: 20px;'>
        <img src='img.png' width='320' alt='AI Powered Segmentation'/>
        <p style='margin-top: 5px; font-style: italic;'>AI Powered Segmentation</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ================= Sidebar Options =================
st.sidebar.markdown("### ⚙️ Segmentation Settings")
st.sidebar.markdown("---")

with st.sidebar.expander("🎨 Quality Settings", expanded=True):
    tta_toggle = st.checkbox("🔄 High Quality Mode (TTA)", True, help="Uses Test-Time Augmentation for better results")
    blur_strength = st.slider("✨ Edge Smoothness", 3, 21, 7, step=2, help="Higher values = softer edges")
    dilate_iter = st.slider("📏 Mask Expansion", 0, 5, 1, help="Expand or shrink the mask")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎭 Background Settings")

use_transparent = st.sidebar.checkbox("🔲 Transparent Background", help="Create PNG with transparency")
bg_type = st.sidebar.radio("Background Style", ["Solid Color","Gradient","Custom Image","Blur Background"])

gradient_colors = None
bg_blur = False
blur_amount = 21

if bg_type == "Solid Color":
    bg_color = st.sidebar.color_picker("🎨 Background Color", "#000000")
    bg_tuple = tuple(int(bg_color.lstrip('#')[i:i+2],16) for i in (0,2,4))
    custom_bg = None
elif bg_type == "Custom Image":
    bg_file = st.sidebar.file_uploader("📸 Upload Background Image", type=["jpg","png"])
    custom_bg = Image.open(bg_file).convert("RGB") if bg_file else None
    bg_tuple = (0,0,0)
elif bg_type == "Gradient":
    col1_exp = st.sidebar.columns(2)
    with col1_exp[0]:
        color1 = st.color_picker("Start", "#000000")
    with col1_exp[1]:
        color2 = st.color_picker("End", "#ff0000")
    col1 = np.array([int(color1.lstrip('#')[i:i+2], 16) for i in (0,2,4)], dtype=np.uint8)
    col2 = np.array([int(color2.lstrip('#')[i:i+2], 16) for i in (0,2,4)], dtype=np.uint8)
    gradient_colors = (col1, col2)
    custom_bg = None
    bg_tuple = (0,0,0)
elif bg_type == "Blur Background":
    bg_blur = True
    blur_amount = st.sidebar.slider("🌫️ Blur Intensity", 5, 99, 21, step=2, help="Higher values = more blur")
    custom_bg = None
    bg_tuple = (0,0,0)

# ================= Artistic Effects Section =================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Artistic Effects")

with st.sidebar.expander("🖌️ Apply Artistic Style", expanded=False):
    artistic_effect = st.selectbox(
        "Effect Type",
        ["None", "Sketch", "Cartoon", "Oil Painting", "Watercolor", "Colored Pencil"],
        help="Apply artistic effects to the segmented object"
    )
    
    if artistic_effect != "None":
        effect_intensity = st.slider(
            "🎚️ Effect Intensity",
            0.0, 1.0, 0.8, 0.1,
            help="Control the strength of the artistic effect"
        )
    else:
        effect_intensity = 1.0

# ================= NEW: Edge Overlay Settings =================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🖍️ Edge Overlay Settings")

with st.sidebar.expander("✏️ Edge Visualization", expanded=False):
    show_edges = st.checkbox("Show Edge Overlay", value=False, help="Display object boundaries")
    
    if show_edges:
        edge_color = st.color_picker("Edge Color", "#00FF00", help="Choose the color for edge lines")
        edge_thickness = st.slider("Edge Thickness", 1, 10, 2, help="Thickness of edge lines")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** High Quality Mode works best for complex objects!")

# ================= Mode Selection =================
st.markdown("### 🖼️ Select Processing Mode")
mode = st.radio("Select Processing Mode", ["Single Image", "Batch Processing"], horizontal=True, label_visibility="collapsed")

# ================= Single Image =================
if mode == "Single Image":
    st.markdown("---")
    uploaded_file = st.file_uploader("📤 Upload Your Image", type=["jpg","jpeg","png"], help="Supported formats: JPG, PNG")
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(img, caption="📷 Uploaded Image", use_container_width=True)

        st.markdown("---")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
        with col_btn2:
            process_btn = st.button("✨ Process Image", use_container_width=True)
        
        if process_btn:
            progress = st.progress(0, text="🔄 Initializing AI model...")
            
            with st.spinner("🎨 AI is working its magic..."):
                for i in range(0,101,20):
                    if i == 20:
                        progress.progress(i, text="🔍 Analyzing image...")
                    elif i == 40:
                        progress.progress(i, text="🎯 Detecting objects...")
                    elif i == 60:
                        progress.progress(i, text="✂️ Segmenting...")
                    elif i == 80:
                        progress.progress(i, text="🎨 Refining edges...")
                    else:
                        progress.progress(i, text="✅ Finalizing...")
                    time.sleep(0.2)

                img_tensor, orig_size, _ = preprocess_image(img)
                mask_tensor = tta_predict(img_tensor) if tta_toggle else model(img_tensor)['out']
                pred_mask = postprocess_mask(mask_tensor, orig_size, blur_strength=blur_strength)
                for _ in range(dilate_iter):
                    pred_mask = cv2.dilate(pred_mask, np.ones((3,3),np.uint8), iterations=1)

                # Apply artistic effect to the original image
                if artistic_effect != "None":
                    img_np_styled = apply_artistic_effect(img_np, artistic_effect, effect_intensity)
                else:
                    img_np_styled = img_np

                result = extract_object(img_np_styled, pred_mask, bg_color=bg_tuple, transparent=use_transparent, custom_bg=custom_bg, gradient=gradient_colors, bg_blur=bg_blur, blur_amount=blur_amount)
                
                # Create edge overlay if enabled
                edge_overlay = None
                if show_edges:
                    edge_overlay = create_edge_overlay(img_np, pred_mask, edge_color=edge_color, edge_thickness=edge_thickness)

            progress.empty()
            st.success("✅ Segmentation Complete!")
            st.markdown("---")

            # Update tabs to include edge overlay
            if show_edges:
                tabs = st.tabs(["🖼️ Side by Side", "↔️ Interactive Comparison", "✏️ Edge Overlay"])
            else:
                tabs = st.tabs(["🖼️ Side by Side", "↔️ Interactive Comparison"])
            
            with tabs[0]:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📷 Original**")
                    st.image(img_np, use_container_width=True)
                with col2:
                    st.markdown("**✨ Segmented**")
                    st.image(result, use_container_width=True)

                st.markdown("---")
                st.markdown("### 📥 Download Results")
                
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    buf_seg = BytesIO()
                    Image.fromarray(result).save(buf_seg, format="PNG")
                    st.download_button("💾 Download Segmented", buf_seg.getvalue(), "segmented_image.png", "image/png", use_container_width=True)
                
                with dl_col2:
                    if use_transparent:
                        buf_trans = BytesIO()
                        Image.fromarray(result).save(buf_trans, format="PNG")
                        st.download_button("💾 Download Transparent", buf_trans.getvalue(), "transparent.png", "image/png", use_container_width=True)

            with tabs[1]:
                img_small = cv2.resize(img_np, (500, int(500*img_np.shape[0]/img_np.shape[1])))
                result_bg = extract_object(img_np_styled, pred_mask, bg_color=bg_tuple, transparent=False, custom_bg=custom_bg, gradient=gradient_colors, bg_blur=bg_blur, blur_amount=blur_amount)
                overlay_small = cv2.resize(result_bg, (500, int(500*result_bg.shape[0]/result_bg.shape[1])))
                image_comparison(img1=img_small, img2=overlay_small, label1="Original", label2="Segmented")
            
            # Edge Overlay Tab
            if show_edges:
                with tabs[2]:
                    st.markdown("**✏️ Edge Boundaries Overlay**")
                    st.image(edge_overlay, caption="Edges Overlay", use_container_width=True)
                    
                    st.markdown("---")
                    buf_edge = BytesIO()
                    Image.fromarray(edge_overlay).save(buf_edge, format="PNG")
                    st.download_button("💾 Download Edge Overlay", buf_edge.getvalue(), "edge_overlay.png", "image/png", use_container_width=True)

# ================= Multiple Images =================
elif mode == "Batch Processing":
    st.markdown("---")
    uploaded_files = st.file_uploader("📤 Upload Multiple Images", type=["jpg","jpeg","png"], accept_multiple_files=True, help="Process multiple images at once")
    
    if uploaded_files:
        st.markdown(f"### 📸 Uploaded {len(uploaded_files)} images")
        
        images = [np.array(Image.open(f).convert("RGB")) for f in uploaded_files]
        cols = st.columns(min(len(images), 4))
        for idx, (col, img_np) in enumerate(zip(cols, images[:4])):
            col.image(img_np, caption=f"Image {idx+1}", use_container_width=True)
        
        if len(images) > 4:
            st.info(f"📋 {len(images)-4} more images ready for processing...")

        st.markdown("---")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
        with col_btn2:
            process_all = st.button("✨ Process All Images", use_container_width=True)
        
        if process_all:
            results = []
            edge_overlays = []
            progress = st.progress(0, text="🚀 Starting batch processing...")
            
            for i, img_np in enumerate(images):
                progress.progress(int((i)/len(images)*100), text=f"🎨 Processing image {i+1} of {len(images)}...")
                
                img_tensor, orig_size, _ = preprocess_image(Image.fromarray(img_np))
                mask_tensor = tta_predict(img_tensor) if tta_toggle else model(img_tensor)['out']
                pred_mask = postprocess_mask(mask_tensor, orig_size, blur_strength=blur_strength)
                for _ in range(dilate_iter):
                    pred_mask = cv2.dilate(pred_mask, np.ones((3,3),np.uint8), iterations=1)
                
                # Apply artistic effect to the original image
                if artistic_effect != "None":
                    img_np_styled = apply_artistic_effect(img_np, artistic_effect, effect_intensity)
                else:
                    img_np_styled = img_np
                
                result = extract_object(img_np_styled, pred_mask, bg_color=bg_tuple, transparent=use_transparent, custom_bg=custom_bg, gradient=gradient_colors, bg_blur=bg_blur, blur_amount=blur_amount)
                results.append(result)
                
                # Create edge overlay if enabled
                if show_edges:
                    edge_overlay = create_edge_overlay(img_np, pred_mask, edge_color=edge_color, edge_thickness=edge_thickness)
                    edge_overlays.append(edge_overlay)
            
            progress.progress(100, text="✅ All images processed!")
            time.sleep(0.5)
            progress.empty()
            
            st.success(f"✅ Successfully processed {len(results)} images!")
            st.markdown("---")
            st.markdown("### 🎉 Results")

            cols = st.columns(min(len(results), 4))
            for idx, (col, result) in enumerate(zip(cols, results[:4])):
                col.image(result, caption=f"Result {idx+1}", use_container_width=True)
            
            if len(results) > 4:
                with st.expander(f"👁️ View all {len(results)} results"):
                    remaining_cols = st.columns(4)
                    for idx, result in enumerate(results[4:], 5):
                        remaining_cols[(idx-5) % 4].image(result, caption=f"Result {idx}", use_container_width=True)
            
            # Show edge overlays if enabled
            if show_edges and edge_overlays:
                st.markdown("---")
                st.markdown("### ✏️ Edge Overlays")
                
                edge_cols = st.columns(min(len(edge_overlays), 4))
                for idx, (col, edge_img) in enumerate(zip(edge_cols, edge_overlays[:4])):
                    col.image(edge_img, caption=f"Edges {idx+1}", use_container_width=True)
                
                if len(edge_overlays) > 4:
                    with st.expander(f"👁️ View all {len(edge_overlays)} edge overlays"):
                        remaining_edge_cols = st.columns(4)
                        for idx, edge_img in enumerate(edge_overlays[4:], 5):
                            remaining_edge_cols[(idx-5) % 4].image(edge_img, caption=f"Edges {idx}", use_container_width=True)

            st.markdown("---")
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for idx, result in enumerate(results):
                    buf = BytesIO()
                    Image.fromarray(result).save(buf, format="PNG")
                    zip_file.writestr(f"segmented_{idx+1}.png", buf.getvalue())
                
                # Add edge overlays to zip if enabled
                if show_edges and edge_overlays:
                    for idx, edge_img in enumerate(edge_overlays):
                        buf = BytesIO()
                        Image.fromarray(edge_img).save(buf, format="PNG")
                        zip_file.writestr(f"edge_overlay_{idx+1}.png", buf.getvalue())
            
            col_dl1, col_dl2, col_dl3 = st.columns([1,1,1])
            with col_dl2:
                st.download_button("📦 Download All as ZIP", zip_buffer.getvalue(), "segmented_images.zip", "application/zip", use_container_width=True)