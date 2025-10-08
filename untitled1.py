import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from io import BytesIO
from streamlit_image_comparison import image_comparison
from skimage import measure
import zipfile
import time
import os

# Try to import segmentation_models_pytorch
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    st.warning("⚠️ segmentation-models-pytorch not installed. Only torchvision models will be available.")

# ================= Device =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= Enhanced Custom CSS with Animations =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --transition-speed: 0.4s;
    --shadow-light: 0 8px 32px rgba(0,0,0,0.08);
    --shadow-medium: 0 12px 48px rgba(0,0,0,0.12);
    --shadow-heavy: 0 20px 60px rgba(0,0,0,0.15);
    --glow: 0 0 20px rgba(230,57,70,0.3);
}

* {
    transition: background-color var(--transition-speed) ease,
                color var(--transition-speed) ease,
                border-color var(--transition-speed) ease,
                box-shadow var(--transition-speed) ease,
                transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
}

@media (prefers-color-scheme: light) {
    .stApp { 
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 50%, #f1f3f5 100%);
        color: #212529;
        font-family: 'Inter', sans-serif;
    }
    
    .hero { 
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 2px solid transparent;
        background-clip: padding-box;
        box-shadow: var(--shadow-medium), 0 0 0 1px rgba(230,57,70,0.1);
        border-radius: 28px;
        padding: 3.5rem 2rem;
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
        background: radial-gradient(circle, rgba(230,57,70,0.08) 0%, transparent 70%);
        animation: pulse 8s ease-in-out infinite;
    }
    
    .hero::after {
        content: '✨';
        position: absolute;
        font-size: 4rem;
        opacity: 0.1;
        top: 20px;
        right: 40px;
        animation: float 6s ease-in-out infinite;
    }
    
    .hero h1 { 
        background: linear-gradient(135deg, #e63946 0%, #ff6b6b 50%, #ff8787 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
        position: relative;
        z-index: 1;
        animation: slideInDown 0.8s ease-out;
    }
    
    .hero p { 
        color: #6c757d;
        font-size: 1.3rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
        letter-spacing: 0.02em;
        animation: slideInUp 0.8s ease-out;
    }
    
    .feature-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: var(--shadow-light);
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        cursor: pointer;
    }
    
    .feature-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: var(--shadow-heavy), var(--glow);
        border-color: rgba(230,57,70,0.3);
    }
    
    .stat-badge {
        background: linear-gradient(135deg, #e63946 0%, #ff6b6b 100%);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(230,57,70,0.3);
        animation: bounceIn 0.6s ease-out;
    }
}

@media (prefers-color-scheme: dark) {
    .stApp { 
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f0f 100%);
        color: #e9ecef;
    }
    
    .hero { 
        background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%);
        border: 2px solid transparent;
        box-shadow: 0 12px 48px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,107,107,0.15);
        border-radius: 28px;
        padding: 3.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero::before {
        background: radial-gradient(circle, rgba(255,107,107,0.12) 0%, transparent 70%);
    }
    
    .hero::after {
        content: '✨';
        position: absolute;
        font-size: 4rem;
        opacity: 0.15;
        top: 20px;
        right: 40px;
        animation: float 6s ease-in-out infinite;
    }
    
    .hero h1 { 
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8787 50%, #ffa5a5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 900;
        animation: slideInDown 0.8s ease-out;
    }
    
    .hero p { 
        color: #adb5bd;
        font-size: 1.3rem;
        animation: slideInUp 0.8s ease-out;
    }
    
    .feature-card {
        background: #1a1a1a;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        border: 1px solid rgba(255,255,255,0.05);
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    .feature-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(0,0,0,0.6), 0 0 30px rgba(255,107,107,0.3);
        border-color: rgba(255,107,107,0.3);
    }
    
    .stat-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8787 100%);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(255,107,107,0.4);
        animation: bounceIn 0.6s ease-out;
    }
}

@keyframes pulse {
    0%, 100% { transform: scale(1) rotate(0deg); opacity: 1; }
    50% { transform: scale(1.1) rotate(5deg); opacity: 0.8; }
}

@keyframes float {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-15px) rotate(10deg); }
}

@keyframes slideInDown {
    from { opacity: 0; transform: translateY(-30px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes bounceIn {
    0% { transform: scale(0.3); opacity: 0; }
    50% { transform: scale(1.05); }
    70% { transform: scale(0.9); }
    100% { transform: scale(1); opacity: 1; }
}

@keyframes shimmer {
    0% { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}

.stButton > button {
    background: linear-gradient(135deg, #e63946 0%, #ff6b6b 100%);
    color: white !important;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 1rem 2.5rem;
    border: none;
    box-shadow: 0 8px 28px rgba(230,57,70,0.35);
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
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    transition: left 0.6s ease;
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button:hover {
    transform: translateY(-4px) scale(1.03);
    background: linear-gradient(135deg, #ff6b6b 0%, #ff8787 100%);
    box-shadow: 0 15px 40px rgba(230,57,70,0.5);
    color: white !important;
}

.stButton > button:active {
    transform: translateY(-1px) scale(0.98);
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #20c997 0%, #38d9a9 100%);
    color: white !important;
    border-radius: 16px;
    font-weight: 600;
    padding: 0.8rem 2rem;
    border: none;
    box-shadow: 0 6px 20px rgba(32,201,151,0.35);
}

.stDownloadButton > button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 10px 30px rgba(32,201,151,0.45);
    background: linear-gradient(135deg, #38d9a9 0%, #51cf66 100%);
}

img {
    border-radius: 20px;
    box-shadow: var(--shadow-medium);
    transition: transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.4s ease;
}

img:hover {
    transform: scale(1.03) rotate(1deg);
    box-shadow: var(--shadow-heavy);
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #e63946, #ff6b6b, #ff8787, #ff6b6b, #e63946);
    background-size: 300% 100%;
    animation: shimmer 2s linear infinite;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: transparent;
    border-bottom: 2px solid rgba(0,0,0,0.05);
}

.stTabs [data-baseweb="tab"] { 
    border-radius: 16px 16px 0 0;
    font-weight: 700;
    padding: 14px 28px;
    border: none;
    font-size: 1.05rem;
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.stTabs [data-baseweb="tab"]:hover {
    transform: translateY(-3px);
}

.stTabs [aria-selected="true"] { 
    background: linear-gradient(135deg, #e63946 0%, #ff6b6b 100%) !important;
    color: white !important;
    box-shadow: 0 8px 25px rgba(230,57,70,0.4) !important;
    transform: translateY(-3px);
}

[data-testid="stFileUploader"] {
    border-radius: 20px;
    padding: 2rem;
    border: 3px dashed;
    transition: all 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #e63946;
    transform: scale(1.01);
}

.stSpinner > div {
    border-top-color: #e63946 !important;
    animation: spin 0.8s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

section[data-testid="stSidebar"] .stRadio > label {
    font-weight: 700;
    font-size: 1.1rem;
    margin-bottom: 0.8rem;
}

.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #e63946, #ff6b6b) !important;
}

.stCheckbox > label {
    font-weight: 600;
    font-size: 1rem;
}

.stAlert {
    border-radius: 16px;
    box-shadow: var(--shadow-light);
    border: none;
    animation: slideInUp 0.5s ease-out;
}

.tooltip {
    position: relative;
    display: inline-block;
    cursor: help;
}

.tooltip:hover::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    padding: 8px 12px;
    background: rgba(0,0,0,0.9);
    color: white;
    border-radius: 8px;
    font-size: 0.85rem;
    white-space: nowrap;
    z-index: 1000;
    animation: slideInDown 0.3s ease-out;
}
</style>
""", unsafe_allow_html=True)

# ================= Model Loading Functions =================
@st.cache_resource
def load_deeplab_smp(model_path):
    """Load DeepLabV3+ model from segmentation_models_pytorch"""
    if not SMP_AVAILABLE:
        return None
    
    try:
        model = smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading SMP model: {str(e)}")
        return None

@st.cache_resource
def load_deeplab_torchvision(model_path):
    """Load DeepLabV3 model from torchvision"""
    try:
        from torchvision.models.segmentation import deeplabv3_resnet50
        
        num_classes = 2
        model = deeplabv3_resnet50(weights=None, aux_loss=True)
        
        # Modify classifier
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
    except Exception as e:
        st.error(f"Error loading Torchvision model: {str(e)}")
        return None

def detect_model_type(model_path):
    """Detect which type of model architecture is used"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check for SMP model (DeepLabV3+)
        if any('segmentation_head' in key for key in checkpoint.keys()):
            return 'smp'
        # Check for torchvision model
        elif any('classifier' in key for key in checkpoint.keys()):
            return 'torchvision'
        else:
            return 'unknown'
    except Exception as e:
        st.error(f"Error detecting model type: {str(e)}")
        return 'unknown'

# ================= Enhanced Artistic Effects =================
def apply_sketch_effect(img, intensity=1.0):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inv_gray = 255 - gray
    blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256.0)
    sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    result = cv2.addWeighted(img, 1 - intensity, sketch_rgb, intensity, 0)
    return result.astype(np.uint8)

def apply_cartoon_effect(img, intensity=1.0):
    num_down = 2
    num_bilateral = 7
    img_color = img.copy()
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)
    img_color = cv2.resize(img_color, (img.shape[1], img.shape[0]))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 9, 2)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cartoon = cv2.bitwise_and(img_color, edges)
    result = cv2.addWeighted(img, 1 - intensity, cartoon, intensity, 0)
    return result.astype(np.uint8)

def apply_oil_paint_effect(img, intensity=1.0):
    result = img.copy()
    iterations = int(3 + 2 * intensity)
    for i in range(iterations):
        result = cv2.bilateralFilter(result, 9, 75, 75)
    result = cv2.medianBlur(result, 5)
    hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    result = cv2.addWeighted(img, 1 - intensity, result, intensity, 0)
    return result.astype(np.uint8)

def apply_watercolor_effect(img, intensity=1.0):
    stylized = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
    result = cv2.addWeighted(img, 1 - intensity, stylized, intensity, 0)
    return result.astype(np.uint8)

def apply_pencil_color_effect(img, intensity=1.0):
    _, pencil_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    result = cv2.addWeighted(img, 1 - intensity, pencil_color, intensity, 0)
    return result.astype(np.uint8)

def apply_vintage_effect(img, intensity=1.0):
    """Apply vintage/retro effect with warm tones"""
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia = cv2.transform(img, sepia_filter)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols//2)
    kernel_y = cv2.getGaussianKernel(rows, rows//2)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    mask = np.stack([mask]*3, axis=2)
    
    output = sepia * mask
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    result = cv2.addWeighted(img, 1 - intensity, output, intensity, 0)
    return result.astype(np.uint8)

def apply_neon_glow_effect(img, intensity=1.0):
    """Apply neon glow effect"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    glow = cv2.GaussianBlur(edges_colored, (15, 15), 0)
    glow = cv2.addWeighted(glow, 2, glow, 0, 0)
    
    result = cv2.addWeighted(img, 1.0, glow, intensity, 0)
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_artistic_effect(img, effect_type, intensity=1.0):
    effects = {
        "None": lambda x, i: x,
        "Sketch": apply_sketch_effect,
        "Cartoon": apply_cartoon_effect,
        "Oil Painting": apply_oil_paint_effect,
        "Watercolor": apply_watercolor_effect,
        "Colored Pencil": apply_pencil_color_effect,
        "Vintage": apply_vintage_effect,
        "Neon Glow": apply_neon_glow_effect
    }
    return effects.get(effect_type, lambda x, i: x)(img, intensity)

# ================= Edge Overlay Function =================
def create_edge_overlay(image, mask, edge_color="#00FF00", edge_thickness=2):
    edge_color_rgb = tuple(int(edge_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    if isinstance(mask, np.ndarray):
        mask_resized = Image.fromarray((mask * 255).astype(np.uint8))
    else:
        mask_resized = mask
    
    if mask_resized.size != pil_image.size:
        mask_resized = mask_resized.resize(pil_image.size, Image.NEAREST)
    
    mask_array = np.array(mask_resized)
    contours = measure.find_contours(mask_array, 128)
    
    overlay_edges = pil_image.copy()
    draw = ImageDraw.Draw(overlay_edges)
    
    for contour in contours:
        contour_scaled = contour * (pil_image.size[0] / mask_resized.width)
        contour_points = [tuple(p[::-1]) for p in contour_scaled]
        
        if len(contour_points) > 1:
            draw.line(contour_points, fill=edge_color_rgb, width=edge_thickness)
    
    return np.array(overlay_edges)

# ================= Preprocessing Functions =================
def preprocess_for_smp(img, size=256):
    """Preprocess for SMP DeepLabV3+ model"""
    from torchvision import transforms as T
    
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
    ])
    
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    return transform(img).unsqueeze(0).to(device), img.size

def preprocess_for_torchvision(img, long_side=256):
    """Preprocess for Torchvision DeepLabV3 model"""
    img = np.array(img)
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    new_w, new_h = int(w*scale), int(h*scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    img_tensor = torch.tensor(img_resized/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor.to(device), (h, w)

# ================= TTA Prediction Functions =================
def tta_predict_smp(model, img_tensor):
    """TTA for SMP model"""
    aug_list = [
        lambda x: x,
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(x, [2]),
        lambda x: torch.rot90(x, k=1, dims=[2,3]),
        lambda x: torch.rot90(x, k=3, dims=[2,3])
    ]
    outputs = []
    with torch.no_grad():
        for f in aug_list:
            t = f(img_tensor)
            out = model(t)
            out = torch.sigmoid(out)
            if f != aug_list[0]:
                if f == aug_list[1]: out = torch.flip(out,[3])
                elif f == aug_list[2]: out = torch.flip(out,[2])
                elif f == aug_list[3]: out = torch.rot90(out,k=3,dims=[2,3])
                elif f == aug_list[4]: out = torch.rot90(out,k=1,dims=[2,3])
            outputs.append(out)
    return torch.mean(torch.stack(outputs), dim=0)

def tta_predict_torchvision(model, img_tensor):
    """TTA for Torchvision model"""
    aug_list = [
        lambda x: x,
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(x, [2]),
        lambda x: torch.rot90(x, k=1, dims=[2,3]),
        lambda x: torch.rot90(x, k=3, dims=[2,3])
    ]
    outputs = []
    with torch.no_grad():
        for f in aug_list:
            t = f(img_tensor)
            out = model(t)['out']
            if f != aug_list[0]:
                if f == aug_list[1]: out = torch.flip(out,[3])
                elif f == aug_list[2]: out = torch.flip(out,[2])
                elif f == aug_list[3]: out = torch.rot90(out,k=3,dims=[2,3])
                elif f == aug_list[4]: out = torch.rot90(out,k=1,dims=[2,3])
            outputs.append(out)
    return torch.mean(torch.stack(outputs), dim=0)

# ================= Post-processing Functions =================
def postprocess_mask_smp(mask_tensor, orig_size, blur_strength=5):
    """Post-process mask from SMP model"""
    mask = mask_tensor.squeeze().cpu().detach().numpy()
    
    # Adaptive threshold
    mask = (mask * 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Resize to original size
    mask = cv2.resize(mask, orig_size, interpolation=cv2.INTER_NEAREST)
    
    # Morphological operations
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Smooth edges
    blurred = cv2.GaussianBlur(mask, (blur_strength, blur_strength), 0)
    _, smooth_mask = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
    
    return (smooth_mask // 255).astype(np.uint8)

def postprocess_mask_torchvision(mask_tensor, orig_size, blur_strength=5):
    """Post-process mask from Torchvision model"""
    mask = torch.argmax(mask_tensor, dim=1).squeeze().cpu().numpy()
    mask = cv2.resize(mask.astype(np.uint8), (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)
    
    # Morphological operations
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
    <h1>✨ AI Object Segmentation Studio</h1>
    <p>Transform your images with state-of-the-art AI-powered precision extraction</p>
    <div style='margin-top: 1.5rem;'>
        <span class='stat-badge'>🚀 Multi-Model</span>
        <span class='stat-badge'>🎨 8 Effects</span>
        <span class='stat-badge'>⚡ Real-time</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= Model Selection in Main Area =================
st.markdown("### 🤖 Model Configuration")

model_col1, model_col2 = st.columns([2, 1])

with model_col1:
    model_file = st.file_uploader("📦 Upload Your Trained Model (.pth)", 
                                  type=["pth", "pt"],
                                  help="Upload your trained DeepLabV3 or DeepLabV3+ model")

with model_col2:
    if model_file is not None:
        # Save uploaded model temporarily
        temp_model_path = "temp_model.pth"
        with open(temp_model_path, "wb") as f:
            f.write(model_file.read())
        
        # Detect model type
        model_type = detect_model_type(temp_model_path)
        
        if model_type == 'smp':
            st.success("✅ SMP DeepLabV3+ Detected")
            model = load_deeplab_smp(temp_model_path)
        elif model_type == 'torchvision':
            st.success("✅ Torchvision DeepLabV3 Detected")
            model = load_deeplab_torchvision(temp_model_path)
        else:
            st.error("❌ Unknown model format")
            model = None
    else:
        st.info("👆 Please upload a model to begin")
        model = None
        model_type = None

# ================= Enhanced Sidebar =================
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.markdown("---")
    
    if model is not None:
        st.markdown(f"**🎯 Active Model:** {model_type.upper()}")
        st.markdown("---")
    
    # Quick Presets
    with st.expander("🎯 Quick Presets", expanded=False):
        preset = st.selectbox("Choose a preset", 
                             ["Custom", "High Quality", "Fast", "Artistic", "Professional"])
        
        if preset == "High Quality":
            tta_toggle = True
            blur_strength = 7
            dilate_iter = 2
        elif preset == "Fast":
            tta_toggle = False
            blur_strength = 5
            dilate_iter = 1
        elif preset == "Artistic":
            tta_toggle = True
            blur_strength = 9
            dilate_iter = 1
        elif preset == "Professional":
            tta_toggle = True
            blur_strength = 5
            dilate_iter = 2
        else:  # Custom
            pass
    
    if preset == "Custom":
        with st.expander("🎨 Quality Settings", expanded=True):
            tta_toggle = st.checkbox("🔄 High Quality Mode (TTA)", True, help="Uses Test-Time Augmentation")
            blur_strength = st.slider("✨ Edge Smoothness", 3, 21, 7, step=2)
            dilate_iter = st.slider("📏 Mask Expansion", 0, 5, 1)
    
    st.markdown("---")
    st.markdown("### 🎭 Background Options")
    
    use_transparent = st.checkbox("🔲 Transparent Background", help="Create PNG with alpha channel")
    bg_type = st.radio("Background Style", 
                      ["Solid Color", "Gradient", "Custom Image", "Blur Background"],
                      help="Choose your background type")
    
    gradient_colors = None
    bg_blur = False
    blur_amount = 21
    
    if bg_type == "Solid Color":
        bg_color = st.color_picker("🎨 Background Color", "#000000")
        bg_tuple = tuple(int(bg_color.lstrip('#')[i:i+2],16) for i in (0,2,4))
        custom_bg = None
    elif bg_type == "Custom Image":
        bg_file = st.file_uploader("📸 Upload Background", type=["jpg","png"])
        custom_bg = Image.open(bg_file).convert("RGB") if bg_file else None
        bg_tuple = (0,0,0)
    elif bg_type == "Gradient":
        st.markdown("**Gradient Colors:**")
        col1_exp = st.columns(2)
        with col1_exp[0]:
            color1 = st.color_picker("Top", "#000000")
        with col1_exp[1]:
            color2 = st.color_picker("Bottom", "#ff0000")
        col1 = np.array([int(color1.lstrip('#')[i:i+2], 16) for i in (0,2,4)], dtype=np.uint8)
        col2 = np.array([int(color2.lstrip('#')[i:i+2], 16) for i in (0,2,4)], dtype=np.uint8)
        gradient_colors = (col1, col2)
        custom_bg = None
        bg_tuple = (0,0,0)
    elif bg_type == "Blur Background":
        bg_blur = True
        blur_amount = st.slider("🌫️ Blur Intensity", 5, 99, 21, step=2)
        custom_bg = None
        bg_tuple = (0,0,0)
    
    st.markdown("---")
    st.markdown("### 🎨 Artistic Effects")
    
    with st.expander("🖌️ Apply Style", expanded=False):
        artistic_effect = st.selectbox(
            "Effect Type",
            ["None", "Sketch", "Cartoon", "Oil Painting", "Watercolor", 
             "Colored Pencil", "Vintage", "Neon Glow"],
            help="Transform your segmented object"
        )
        
        if artistic_effect != "None":
            effect_intensity = st.slider("🎚️ Effect Strength", 0.0, 1.0, 0.8, 0.1)
        else:
            effect_intensity = 1.0
    
    st.markdown("---")
    st.markdown("### 🖍️ Edge Visualization")
    
    with st.expander("✏️ Edge Settings", expanded=False):
        show_edges = st.checkbox("Show Edge Overlay", value=False)
        
        if show_edges:
            edge_color = st.color_picker("Edge Color", "#00FF00")
            edge_thickness = st.slider("Edge Thickness", 1, 10, 2)
            edge_style = st.radio("Edge Style", ["Solid", "Glowing"])
    
    st.markdown("---")
    
    # Stats display
    st.markdown("### 📊 Session Stats")
    if 'processed_count' not in st.session_state:
        st.session_state.processed_count = 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Images Processed", st.session_state.processed_count)
    with col2:
        if model_type:
            st.metric("Model Type", model_type.upper())
        else:
            st.metric("Model Type", "None")
    
    st.info("💡 **Pro Tip:** Use High Quality Mode for best results on complex objects!")

# Only show processing options if model is loaded
if model is None:
    st.warning("⚠️ Please upload a trained model to continue")
    st.stop()

# ================= Mode Selection with Enhanced UI =================
st.markdown("### 🖼️ Processing Mode")

mode_cols = st.columns(2)
with mode_cols[0]:
    if st.button("🎯 Single Image", use_container_width=True, key="mode_single"):
        st.session_state.mode = "Single Image"
with mode_cols[1]:
    if st.button("📦 Batch Processing", use_container_width=True, key="mode_batch"):
        st.session_state.mode = "Batch Processing"

if 'mode' not in st.session_state:
    st.session_state.mode = "Single Image"

mode = st.session_state.mode

# ================= Single Image Mode =================
if mode == "Single Image":
    st.markdown("---")
    
    # Enhanced upload section
    st.markdown("""
    <div class='feature-card' style='text-align: center; margin-bottom: 2rem;'>
        <h3 style='margin-bottom: 1rem;'>📤 Upload Your Image</h3>
        <p style='color: #6c757d;'>Drag and drop or click to browse</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"], 
                                     help="Supported: JPG, PNG (Max 200MB)", 
                                     label_visibility="collapsed",
                                     key="single_upload")
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        
        # Display with enhanced layout
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("#### 📷 Original Image")
            st.image(img, use_container_width=True)
            
            # Image info
            info_cols = st.columns(3)
            with info_cols[0]:
                st.metric("Width", f"{img.width}px")
            with info_cols[1]:
                st.metric("Height", f"{img.height}px")
            with info_cols[2]:
                st.metric("Size", f"{uploaded_file.size // 1024}KB")
        
        st.markdown("---")
        
        # Enhanced process button
        col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
        with col_btn2:
            process_btn = st.button("✨ Process Image", use_container_width=True, type="primary")
        
        if process_btn:
            progress_text = st.empty()
            progress = st.progress(0)
            
            progress_stages = [
                ("🔄 Initializing AI model...", 0),
                ("🔍 Analyzing image structure...", 20),
                ("🎯 Detecting object boundaries...", 40),
                ("✂️ Segmenting with precision...", 60),
                ("🎨 Refining edges...", 80),
                ("✅ Finalizing masterpiece...", 95)
            ]
            
            with st.spinner("🎨 AI is crafting your masterpiece..."):
                for text, value in progress_stages:
                    progress_text.markdown(f"**{text}**")
                    progress.progress(value)
                    time.sleep(0.3)
                
                # Process based on model type
                if model_type == 'smp':
                    img_tensor, orig_size = preprocess_for_smp(img)
                    if tta_toggle:
                        mask_tensor = tta_predict_smp(model, img_tensor)
                    else:
                        with torch.no_grad():
                            mask_tensor = torch.sigmoid(model(img_tensor))
                    pred_mask = postprocess_mask_smp(mask_tensor, orig_size, blur_strength=blur_strength)
                
                elif model_type == 'torchvision':
                    img_tensor, orig_size = preprocess_for_torchvision(img)
                    if tta_toggle:
                        mask_tensor = tta_predict_torchvision(model, img_tensor)
                    else:
                        with torch.no_grad():
                            mask_tensor = model(img_tensor)['out']
                    pred_mask = postprocess_mask_torchvision(mask_tensor, orig_size, blur_strength=blur_strength)
                
                # Dilate mask
                for _ in range(dilate_iter):
                    pred_mask = cv2.dilate(pred_mask, np.ones((3,3),np.uint8), iterations=1)
                
                # Apply artistic effect
                if artistic_effect != "None":
                    img_np_styled = apply_artistic_effect(img_np, artistic_effect, effect_intensity)
                else:
                    img_np_styled = img_np
                
                result = extract_object(img_np_styled, pred_mask, bg_color=bg_tuple, 
                                      transparent=use_transparent, custom_bg=custom_bg, 
                                      gradient=gradient_colors, bg_blur=bg_blur, 
                                      blur_amount=blur_amount)
                
                # Create edge overlay if enabled
                edge_overlay = None
                if show_edges:
                    edge_overlay = create_edge_overlay(img_np, pred_mask, 
                                                      edge_color=edge_color, 
                                                      edge_thickness=edge_thickness)
                    if 'edge_style' in locals() and edge_style == "Glowing":
                        edge_overlay = cv2.GaussianBlur(edge_overlay, (5, 5), 0)
                
                progress.progress(100)
                time.sleep(0.3)
                progress.empty()
                progress_text.empty()
                
                st.session_state.processed_count += 1
            
            st.balloons()
            st.success("✅ Segmentation Complete! Your image looks amazing! 🎉")
            st.markdown("---")
            
            # Enhanced results display
            tab_list = ["🖼️ Side by Side", "↔️ Interactive Compare", "🎭 Mask View"]
            if show_edges:
                tab_list.append("✏️ Edge Overlay")
            
            tabs = st.tabs(tab_list)
            
            with tabs[0]:
                st.markdown("### Results Comparison")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📷 Original**")
                    st.image(img_np, use_container_width=True)
                with col2:
                    st.markdown("**✨ Segmented**")
                    st.image(result, use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 📥 Download Options")
                
                dl_cols = st.columns(3)
                with dl_cols[0]:
                    buf_seg = BytesIO()
                    Image.fromarray(result).save(buf_seg, format="PNG")
                    st.download_button("💾 Segmented Image", buf_seg.getvalue(), 
                                     "segmented.png", "image/png", use_container_width=True)
                
                with dl_cols[1]:
                    if use_transparent:
                        buf_trans = BytesIO()
                        Image.fromarray(result).save(buf_trans, format="PNG")
                        st.download_button("💾 Transparent PNG", buf_trans.getvalue(), 
                                         "transparent.png", "image/png", use_container_width=True)
                
                with dl_cols[2]:
                    buf_mask = BytesIO()
                    mask_img = (pred_mask * 255).astype(np.uint8)
                    Image.fromarray(mask_img).save(buf_mask, format="PNG")
                    st.download_button("💾 Mask Only", buf_mask.getvalue(), 
                                     "mask.png", "image/png", use_container_width=True)
            
            with tabs[1]:
                st.markdown("### Interactive Slider Comparison")
                img_small = cv2.resize(img_np, (600, int(600*img_np.shape[0]/img_np.shape[1])))
                result_bg = extract_object(img_np_styled, pred_mask, bg_color=bg_tuple, 
                                          transparent=False, custom_bg=custom_bg, 
                                          gradient=gradient_colors, bg_blur=bg_blur, 
                                          blur_amount=blur_amount)
                overlay_small = cv2.resize(result_bg, (600, int(600*result_bg.shape[0]/result_bg.shape[1])))
                image_comparison(img1=img_small, img2=overlay_small, 
                               label1="Original", label2="Segmented")
            
            with tabs[2]:
                st.markdown("### Segmentation Mask")
                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    mask_display = (pred_mask * 255).astype(np.uint8)
                    mask_colored = cv2.applyColorMap(mask_display, cv2.COLORMAP_JET)
                    mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
                    st.image(mask_colored, caption="Color-mapped Mask", use_container_width=True)
            
            if show_edges and edge_overlay is not None:
                with tabs[3]:
                    st.markdown("### Edge Boundary Visualization")
                    st.image(edge_overlay, caption="Detected Boundaries", use_container_width=True)
                    
                    st.markdown("---")
                    buf_edge = BytesIO()
                    Image.fromarray(edge_overlay).save(buf_edge, format="PNG")
                    col1, col2, col3 = st.columns([1,1,1])
                    with col2:
                        st.download_button("💾 Download Edge Overlay", buf_edge.getvalue(), 
                                         "edges.png", "image/png", use_container_width=True)

# ================= Batch Processing Mode =================
elif mode == "Batch Processing":
    st.markdown("---")
    
    st.markdown("""
    <div class='feature-card' style='text-align: center; margin-bottom: 2rem;'>
        <h3 style='margin-bottom: 1rem;'>📦 Batch Upload</h3>
        <p style='color: #6c757d;'>Process multiple images simultaneously</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("", type=["jpg","jpeg","png"], 
                                     accept_multiple_files=True,
                                     help="Upload multiple images for batch processing",
                                     label_visibility="collapsed",
                                     key="batch_upload")
    
    if uploaded_files:
        st.markdown(f"""
        <div class='stat-badge' style='font-size: 1.1rem; margin: 1rem 0;'>
            📸 {len(uploaded_files)} images ready for processing
        </div>
        """, unsafe_allow_html=True)
        
        images = [np.array(Image.open(f).convert("RGB")) for f in uploaded_files]
        
        # Preview grid
        st.markdown("### 🖼️ Preview Gallery")
        cols = st.columns(min(len(images), 4))
        for idx, (col, img_np) in enumerate(zip(cols, images[:4])):
            with col:
                st.image(img_np, caption=f"Image {idx+1}", use_container_width=True)
        
        if len(images) > 4:
            with st.expander(f"👁️ View all {len(images)} images"):
                preview_cols = st.columns(4)
                for idx, img_np in enumerate(images[4:], 5):
                    with preview_cols[(idx-5) % 4]:
                        st.image(img_np, caption=f"Image {idx}", use_container_width=True)
        
        st.markdown("---")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
        with col_btn2:
            process_all = st.button("✨ Process All Images", use_container_width=True, type="primary")
        
        if process_all:
            results = []
            edge_overlays = []
            
            progress_text = st.empty()
            progress = st.progress(0)
            
            start_time = time.time()
            
            for i, img_np in enumerate(images):
                progress_pct = int((i)/len(images)*100)
                progress_text.markdown(f"**🎨 Processing image {i+1} of {len(images)}... ({progress_pct}%)**")
                progress.progress(progress_pct)
                
                # Process based on model type
                img_pil = Image.fromarray(img_np)
                
                if model_type == 'smp':
                    img_tensor, orig_size = preprocess_for_smp(img_pil)
                    if tta_toggle:
                        mask_tensor = tta_predict_smp(model, img_tensor)
                    else:
                        with torch.no_grad():
                            mask_tensor = torch.sigmoid(model(img_tensor))
                    pred_mask = postprocess_mask_smp(mask_tensor, orig_size, blur_strength=blur_strength)
                
                elif model_type == 'torchvision':
                    img_tensor, orig_size = preprocess_for_torchvision(img_pil)
                    if tta_toggle:
                        mask_tensor = tta_predict_torchvision(model, img_tensor)
                    else:
                        with torch.no_grad():
                            mask_tensor = model(img_tensor)['out']
                    pred_mask = postprocess_mask_torchvision(mask_tensor, orig_size, blur_strength=blur_strength)
                
                # Dilate mask
                for _ in range(dilate_iter):
                    pred_mask = cv2.dilate(pred_mask, np.ones((3,3),np.uint8), iterations=1)
                
                # Apply artistic effect
                if artistic_effect != "None":
                    img_np_styled = apply_artistic_effect(img_np, artistic_effect, effect_intensity)
                else:
                    img_np_styled = img_np
                
                result = extract_object(img_np_styled, pred_mask, bg_color=bg_tuple, 
                                      transparent=use_transparent, custom_bg=custom_bg, 
                                      gradient=gradient_colors, bg_blur=bg_blur, 
                                      blur_amount=blur_amount)
                results.append(result)
                
                if show_edges:
                    edge_overlay = create_edge_overlay(img_np, pred_mask, 
                                                      edge_color=edge_color, 
                                                      edge_thickness=edge_thickness)
                    edge_overlays.append(edge_overlay)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            progress.progress(100)
            time.sleep(0.3)
            progress.empty()
            progress_text.empty()
            
            st.session_state.processed_count += len(results)
            
            st.balloons()
            st.success(f"✅ Successfully processed {len(results)} images in {processing_time:.1f} seconds! 🎉")
            
            # Performance metrics
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Images Processed", len(results))
            with metric_cols[1]:
                st.metric("Total Time", f"{processing_time:.1f}s")
            with metric_cols[2]:
                st.metric("Avg Time/Image", f"{processing_time/len(results):.1f}s")
            with metric_cols[3]:
                st.metric("Quality Mode", "High" if tta_toggle else "Fast")
            
            st.markdown("---")
            st.markdown("### 🎉 Batch Results")
            
            # Results grid
            result_cols = st.columns(min(len(results), 4))
            for idx, (col, result) in enumerate(zip(result_cols, results[:4])):
                with col:
                    st.image(result, caption=f"Result {idx+1}", use_container_width=True)
            
            if len(results) > 4:
                with st.expander(f"👁️ View all {len(results)} results"):
                    remaining_cols = st.columns(4)
                    for idx, result in enumerate(results[4:], 5):
                        with remaining_cols[(idx-5) % 4]:
                            st.image(result, caption=f"Result {idx}", use_container_width=True)
            
            # Edge overlays display
            if show_edges and edge_overlays:
                st.markdown("---")
                st.markdown("### ✏️ Edge Overlays")
                
                edge_cols = st.columns(min(len(edge_overlays), 4))
                for idx, (col, edge_img) in enumerate(zip(edge_cols, edge_overlays[:4])):
                    with col:
                        st.image(edge_img, caption=f"Edges {idx+1}", use_container_width=True)
                
                if len(edge_overlays) > 4:
                    with st.expander(f"👁️ View all {len(edge_overlays)} edge overlays"):
                        remaining_edge_cols = st.columns(4)
                        for idx, edge_img in enumerate(edge_overlays[4:], 5):
                            with remaining_edge_cols[(idx-5) % 4]:
                                st.image(edge_img, caption=f"Edges {idx}", use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📦 Download Everything")
            
            # Create comprehensive ZIP
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                # Add segmented images
                for idx, result in enumerate(results):
                    buf = BytesIO()
                    Image.fromarray(result).save(buf, format="PNG")
                    zip_file.writestr(f"segmented/image_{idx+1:03d}.png", buf.getvalue())
                
                # Add edge overlays
                if show_edges and edge_overlays:
                    for idx, edge_img in enumerate(edge_overlays):
                        buf = BytesIO()
                        Image.fromarray(edge_img).save(buf, format="PNG")
                        zip_file.writestr(f"edges/edge_{idx+1:03d}.png", buf.getvalue())
                
                # Add processing report
                report = f"""AI Object Segmentation - Batch Processing Report
================================================================================

PROCESSING SUMMARY
------------------
Total Images Processed: {len(results)}
Processing Time: {processing_time:.2f} seconds
Average Time per Image: {processing_time/len(results):.2f} seconds
Model Type: {model_type.upper()}
Quality Mode: {"High (TTA Enabled)" if tta_toggle else "Fast (TTA Disabled)"}
Device: {"GPU (CUDA)" if torch.cuda.is_available() else "CPU"}

CONFIGURATION SETTINGS
----------------------
Edge Smoothness (Blur): {blur_strength}
Mask Expansion (Dilate): {dilate_iter}
Artistic Effect: {artistic_effect}
Background Type: {bg_type}
Transparent Background: {use_transparent}
Edge Overlay: {show_edges}

QUALITY METRICS
---------------
Model Architecture: {"DeepLabV3+ (SMP)" if model_type == "smp" else "DeepLabV3 (Torchvision)"}
Test-Time Augmentation: {"Enabled (5 augmentations)" if tta_toggle else "Disabled"}
Post-Processing: Morphological operations + Gaussian smoothing

OUTPUT FILES
------------
Segmented Images: segmented/image_001.png to segmented/image_{len(results):03d}.png
{"Edge Overlays: edges/edge_001.png to edges/edge_" + f"{len(edge_overlays):03d}.png" if show_edges and edge_overlays else "Edge Overlays: Not generated"}

NOTES
-----
- All images are saved in PNG format for maximum quality
- Transparent backgrounds preserve alpha channel
- Edge overlays show detected object boundaries
- Processing timestamp: {time.strftime("%Y-%m-%d %H:%M:%S")}

Generated by AI Object Segmentation Studio
================================================================================
"""
                zip_file.writestr("processing_report.txt", report)
            
            zip_buffer.seek(0)
            
            col1, col2, col3 = st.columns([1,1,1])
            with col2:
                st.download_button(
                    label="📥 Download Complete Package (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"segmentation_batch_{len(results)}_images.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            
            st.markdown("---")
            st.info(f"💡 **Package Contents:** {len(results)} segmented images{', ' + str(len(edge_overlays)) + ' edge overlays' if edge_overlays else ''}, and a detailed processing report!")

# ================= Footer =================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #6c757d;'>
    <p style='font-size: 0.9rem; margin-bottom: 0.5rem;'>
        ✨ <strong>AI Object Segmentation Studio</strong> ✨
    </p>
    <p style='font-size: 0.8rem;'>
        Powered by PyTorch • DeepLabV3/V3+ • Advanced Computer Vision
    </p>
    <p style='font-size: 0.75rem; margin-top: 1rem;'>
        💡 Tip: Use High Quality mode for best results | 🚀 Batch process for efficiency
    </p>
</div>
""", unsafe_allow_html=True)