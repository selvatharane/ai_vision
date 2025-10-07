import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image, ImageDraw
from io import BytesIO
from skimage import measure
import segmentation_models_pytorch as smp

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
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
    .stTabs [data-baseweb="tab"] { 
        background: #f1f3f5; color: #495057;
        border-radius: 12px 12px 0 0; font-weight: 600; padding: 12px 24px; border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .stTabs [data-baseweb="tab"]:hover { background: #e9ecef; transform: translateY(-2px); }
    .stTabs [aria-selected="true"] { 
        background: linear-gradient(135deg, #e63946 0%, #ff6b6b 100%) !important; color: white !important;
        box-shadow: 0 6px 20px rgba(230,57,70,0.35) !important; transform: translateY(-2px);
    }
    .upload-section { background: white; border-radius: 16px; padding: 2rem; box-shadow: var(--shadow-light); border: 2px dashed #dee2e6; transition: all 0.3s ease; }
    .upload-section:hover { border-color: #e63946; box-shadow: var(--shadow-medium); }
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
    .stApp { 
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f0f 100%);
        color: #e9ecef; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    .hero { 
        background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%);
        border: 1px solid rgba(255,107,107,0.15); box-shadow: 0 12px 48px rgba(0,0,0,0.6);
        border-radius: 24px; padding: 3rem 2rem; text-align: center; margin-bottom: 2rem;
        position: relative; overflow: hidden;
    }
    .hero::before {
        content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,107,107,0.08) 0%, transparent 70%);
        animation: pulse 8s ease-in-out infinite;
    }
    .hero h1 { 
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8787 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem; letter-spacing: -0.02em; position: relative; z-index: 1;
    }
    .hero p { color: #adb5bd; font-size: 1.2rem; font-weight: 400; position: relative; z-index: 1; letter-spacing: 0.01em; }
    section[data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #1a1a1a 0%, #0f0f0f 100%);
        border-right: 1px solid rgba(255,255,255,0.08); box-shadow: 4px 0 24px rgba(0,0,0,0.5);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
    .stTabs [data-baseweb="tab"] { background: #2a2a2a; color: #adb5bd; border-radius: 12px 12px 0 0; font-weight: 600; padding: 12px 24px; border: none; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
    .stTabs [data-baseweb="tab"]:hover { background: #333333; transform: translateY(-2px); }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #ff6b6b 0%, #ff8787 100%) !important; color: white !important; box-shadow: 0 6px 20px rgba(255,107,107,0.4) !important; transform: translateY(-2px); }
    .upload-section { background: #1a1a1a; border-radius: 16px; padding: 2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.4); border: 2px dashed #333333; transition: all 0.3s ease; }
    .upload-section:hover { border-color: #ff6b6b; box-shadow: 0 12px 48px rgba(0,0,0,0.6); }
}

@keyframes pulse { 0%,100%{transform:scale(1) rotate(0deg); opacity:1;} 50%{transform:scale(1.1) rotate(5deg); opacity:0.8;} }
@keyframes float { 0%,100%{transform:translateY(0px);} 50%{transform:translateY(-10px);} }
@keyframes shimmer { 0%{background-position:-1000px 0;} 100%{background-position:1000px 0;} }

/* Buttons, download, images, sliders, etc. - included exactly as given in your code */
.stButton > button { background: linear-gradient(135deg, #e63946 0%, #ff6b6b 100%); color:white !important; border-radius:16px; font-weight:700; font-size:1.05rem; padding:0.85rem 2rem; border:none; box-shadow:0 6px 24px rgba(230,57,70,0.3); position:relative; overflow:hidden; letter-spacing:0.02em;}
.stButton > button::before { content:''; position:absolute; top:0; left:-100%; width:100%; height:100%; background:linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent); transition:left 0.5s ease;}
.stButton > button:hover::before { left:100%; }
.stButton > button:hover, .stButton > button:focus, .stButton > button:active { transform:translateY(-3px) scale(1.02); background:linear-gradient(135deg, #ff6b6b 0%, #ff8787 100%); box-shadow:0 12px 32px rgba(230,57,70,0.45); color:white !important;}
.stDownloadButton > button { background:linear-gradient(135deg,#20c997 0%,#38d9a9 100%); color:white !important; border-radius:14px; font-weight:600; padding:0.7rem 1.8rem; border:none; box-shadow:0 4px 16px rgba(32,201,151,0.3);}
.stDownloadButton > button:hover { transform:translateY(-2px); box-shadow:0 8px 24px rgba(32,201,151,0.4); background:linear-gradient(135deg,#38d9a9 0%,#51cf66 100%); color:white !important;}
img { border-radius:16px; box-shadow:var(--shadow-medium); transition: transform 0.3s ease, box-shadow 0.3s ease; }
img:hover { transform: scale(1.02); box-shadow: var(--shadow-heavy);}
</style>
""", unsafe_allow_html=True)

# ================= Device =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= Load Model =================
@st.cache_resource
def load_model():
    num_classes = 1
    model_path = "best_deeplab.pth"
    model = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

model = load_model()

# ================= TTA Prediction, Postprocessing, Extraction =================
def tta_predict(img_tensor):
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
            if isinstance(out, dict): out = out['out']
            if f != aug_list[0]:
                if f == aug_list[1]: out = torch.flip(out,[3])
                elif f == aug_list[2]: out = torch.flip(out,[2])
                elif f == aug_list[3]: out = torch.rot90(out,k=3,dims=[2,3])
                elif f == aug_list[4]: out = torch.rot90(out,k=1,dims=[2,3])
            outputs.append(out)
    return torch.mean(torch.stack(outputs), dim=0)

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

def extract_object(img, mask, transparent=False):
    mask = mask.astype(np.float32)
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    mask = np.expand_dims(mask, axis=-1)
    mask = mask / (mask.max() + 1e-8)
    if transparent:
        rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA).astype(np.float32)
        rgba[..., 3] = (mask.squeeze() * 255).astype(np.uint8)
        return rgba.astype(np.uint8)
    else:
        return (img.astype(np.float32) * mask + 0 * (1 - mask)).astype(np.uint8)

def create_edge_overlay(image, mask, edge_color="#00FF00", edge_thickness=2):
    edge_color_rgb = tuple(int(edge_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    mask_resized = Image.fromarray((mask * 255).astype(np.uint8))
    if mask_resized.size != pil_image.size: mask_resized = mask_resized.resize(pil_image.size, Image.NEAREST)
    mask_array = np.array(mask_resized)
    contours = measure.find_contours(mask_array, 128)
    overlay_edges = pil_image.copy()
    draw = ImageDraw.Draw(overlay_edges)
    for contour in contours:
        contour_points = [tuple(p[::-1]) for p in contour]
        if len(contour_points) > 1: draw.line(contour_points, fill=edge_color_rgb, width=edge_thickness)
    return np.array(overlay_edges)

# ================= Streamlit UI =================
st.title("✨ AI Object Segmentation")
uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
use_tta = st.checkbox("High Quality Mode (TTA)", True)
transparent_bg = st.checkbox("Transparent Background", False)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    
    img_tensor, orig_size, _ = preprocess_image(img)
    mask_tensor = tta_predict(img_tensor) if use_tta else model(img_tensor)['out']
    pred_mask = postprocess_mask(mask_tensor, orig_size)
    
    result = extract_object(img_np, pred_mask, transparent=transparent_bg)
    edge_overlay = create_edge_overlay(img_np, pred_mask)
    
    st.image(result, caption="Segmented Result", use_container_width=True)
    st.image(edge_overlay, caption="Edge Overlay", use_container_width=True)
    
    buf = BytesIO()
    Image.fromarray(result).save(buf, format="PNG")
    st.download_button("Download Segmented", buf.getvalue(), "segmented.png", "image/png")
