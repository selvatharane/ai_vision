import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image, ImageDraw
from io import BytesIO
from streamlit_image_comparison import image_comparison
from skimage import measure
import segmentation_models_pytorch as smp
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
def load_model():
    import os
    num_classes = 2
    model_path = "best_deeplab.pth"
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None

    # Using SMP DeepLabV3 with ResNet34 backbone
    model = smp.DeepLabV3(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=num_classes
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Safe loading
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

# ================= Artistic Effects Functions =================
def apply_sketch_effect(img, intensity=1.0):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inv_gray = 255 - gray
    blur = cv2.GaussianBlur(inv_gray, (21,21),0)
    sketch = cv2.divide(gray, 255 - blur, scale=256.0)
    sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(img,1-intensity,sketch_rgb,intensity,0).astype(np.uint8)

def apply_cartoon_effect(img, intensity=1.0):
    num_down, num_bilateral = 2, 7
    img_color = img.copy()
    for _ in range(num_down): img_color = cv2.pyrDown(img_color)
    for _ in range(num_bilateral): img_color = cv2.bilateralFilter(img_color, 9,9,7)
    for _ in range(num_down): img_color = cv2.pyrUp(img_color)
    img_color = cv2.resize(img_color,(img.shape[1],img.shape[0]))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray,7)
    edges = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,2)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cartoon = cv2.bitwise_and(img_color,edges)
    return cv2.addWeighted(img,1-intensity,cartoon,intensity,0).astype(np.uint8)

def apply_oil_paint_effect(img,intensity=1.0):
    result = img.copy()
    for _ in range(int(3+2*intensity)):
        result = cv2.bilateralFilter(result,9,75,75)
    result = cv2.medianBlur(result,5)
    hsv = cv2.cvtColor(result,cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1]*1.3,0,255)
    result = cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2RGB)
    return cv2.addWeighted(img,1-intensity,result,intensity,0).astype(np.uint8)

def apply_watercolor_effect(img,intensity=1.0):
    stylized = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
    return cv2.addWeighted(img,1-intensity,stylized,intensity,0).astype(np.uint8)

def apply_pencil_color_effect(img,intensity=1.0):
    _, pencil_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return cv2.addWeighted(img,1-intensity,pencil_color,intensity,0).astype(np.uint8)

def apply_artistic_effect(img,effect_type,intensity=1.0):
    if effect_type=="None": return img
    elif effect_type=="Sketch": return apply_sketch_effect(img,intensity)
    elif effect_type=="Cartoon": return apply_cartoon_effect(img,intensity)
    elif effect_type=="Oil Painting": return apply_oil_paint_effect(img,intensity)
    elif effect_type=="Watercolor": return apply_watercolor_effect(img,intensity)
    elif effect_type=="Colored Pencil": return apply_pencil_color_effect(img,intensity)
    return img

# ================= Edge Overlay Function =================
def create_edge_overlay(image, mask, edge_color="#00FF00", edge_thickness=2):
    edge_color_rgb = tuple(int(edge_color.lstrip('#')[i:i+2],16) for i in (0,2,4))
    if isinstance(image,np.ndarray): pil_image = Image.fromarray(image)
    else: pil_image = image
    if isinstance(mask,np.ndarray): mask_resized = Image.fromarray((mask*255).astype(np.uint8))
    else: mask_resized = mask
    if mask_resized.size != pil_image.size: mask_resized = mask_resized.resize(pil_image.size,Image.NEAREST)
    mask_array = np.array(mask_resized)
    contours = measure.find_contours(mask_array,128)
    overlay_edges = pil_image.copy()
    draw = ImageDraw.Draw(overlay_edges)
    for contour in contours:
        contour_points = [tuple(p[::-1]) for p in contour]
        if len(contour_points)>1: draw.line(contour_points, fill=edge_color_rgb, width=edge_thickness)
    return np.array(overlay_edges)

# ================= Preprocess =================
def preprocess_image(img,long_side=256):
    img = np.array(img)
    h,w = img.shape[:2]
    scale = long_side/max(h,w)
    new_w,new_h = int(w*scale),int(h*scale)
    img_resized = cv2.resize(img,(new_w,new_h))
    img_tensor = torch.tensor(img_resized/255.,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    img_tensor = (img_tensor-mean)/std
    return img_tensor.to(device),(h,w),img

# ================= TTA Prediction =================
def tta_predict(img_tensor):
    aug_list = [
        lambda x:x,
        lambda x:torch.flip(x,[3]),
        lambda x:torch.flip(x,[2]),
        lambda x:torch.rot90(x,k=1,dims=[2,3]),
        lambda x:torch.rot90(x,k=3,dims=[2,3])
    ]
    outputs = []
    with torch.no_grad():
        for f in aug_list:
            t = f(img_tensor)
            out = model(t)
            if isinstance(out, dict):  # SMP returns dict with 'out'
                out = out['out']
            if f != aug_list[0]:
                if f == aug_list[1]: out = torch.flip(out,[3])
                elif f == aug_list[2]: out = torch.flip(out,[2])
                elif f == aug_list[3]: out = torch.rot90(out,k=3,dims=[2,3])
                elif f == aug_list[4]: out = torch.rot90(out,k=1,dims=[2,3])
            outputs.append(out)
    return torch.mean(torch.stack(outputs),dim=0)

# ================= Post-process mask =================
def postprocess_mask(mask_tensor, orig_size, blur_strength=5):
    mask = torch.argmax(mask_tensor, dim=1).squeeze().cpu().numpy()
    mask = cv2.resize(mask.astype(np.uint8),(orig_size[1],orig_size[0]),interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask = cv2.dilate(mask,kernel,iterations=1)
    mask = mask.astype(np.uint8)*255
    blurred = cv2.GaussianBlur(mask,(blur_strength,blur_strength),0)
    _, smooth_mask = cv2.threshold(blurred,128,255,cv2.THRESH_BINARY)
    return (smooth_mask//255).astype(np.uint8)

# ================= Extract Object =================
def extract_object(img, mask, bg_color=(0,0,0), transparent=False, custom_bg=None, gradient=None, bg_blur=False, blur_amount=21):
    mask = mask.astype(np.float32)
    mask = cv2.GaussianBlur(mask,(7,7),0)
    mask = np.expand_dims(mask,axis=-1)/(mask.max()+1e-8)
    if transparent:
        rgba = cv2.cvtColor(img,cv2.COLOR_RGB2RGBA).astype(np.float32)
        rgba[...,3] = (mask.squeeze()*255).astype(np.uint8)
        return rgba.astype(np.uint8)
    else:
        if bg_blur:
            bg_resized = cv2.GaussianBlur(img,(blur_amount,blur_amount),0)
        elif gradient is not None:
            bg_resized = gradient
        elif custom_bg is not None:
            bg_resized = cv2.resize(custom_bg,(img.shape[1],img.shape[0]))
        else:
            bg_resized = np.ones_like(img)*np.array(bg_color,dtype=np.uint8)
        result = img*mask + bg_resized*(1-mask)
        return result.astype(np.uint8)

# ================= Streamlit UI =================
st.title("AI Vision Object Extraction")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_tensor, orig_size, orig_img = preprocess_image(image)
    
    tta_toggle = st.checkbox("Use TTA for better accuracy", value=True)
    artistic_effect = st.selectbox("Artistic Effect", ["None","Sketch","Cartoon","Oil Painting","Watercolor","Colored Pencil"])
    edge_overlay_toggle = st.checkbox("Overlay Edges", value=True)
    edge_color = st.color_picker("Edge Color","#00FF00")
    
    with st.spinner("Processing..."):
        mask_tensor = tta_predict(img_tensor) if tta_toggle else model(img_tensor)['out']
        mask = postprocess_mask(mask_tensor, orig_size)
        result_img = apply_artistic_effect(orig_img, artistic_effect)
        if edge_overlay_toggle:
            result_img = create_edge_overlay(result_img, mask, edge_color=edge_color)
        st.image(result_img, use_column_width=True)
        st.success("Done!")
