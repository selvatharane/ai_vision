import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image, ImageDraw
import numpy as np
import cv2
from skimage import measure
from io import BytesIO
import zipfile
import time

# =========================
# Page Config
# =========================
st.set_page_config(page_title="AI Vision Extractor", layout="wide", page_icon="🎨", initial_sidebar_state="expanded")

# =========================
# Custom Artistic CSS
# =========================
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

/* Global transitions */
* {
    transition: background-color var(--transition-speed) ease,
                color var(--transition-speed) ease,
                border-color var(--transition-speed) ease,
                box-shadow var(--transition-speed) ease,
                transform 0.3s ease;
}

/* Light & Dark Mode Styling */
@media (prefers-color-scheme: light) {
    .stApp { 
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 50%, #f1f3f5 100%);
        color: #212529;
        font-family: 'Inter', sans-serif;
    }
}
@media (prefers-color-scheme: dark) {
    .stApp { 
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f0f 100%);
        color: #e9ecef;
        font-family: 'Inter', sans-serif;
    }
}

/* Hero Section */
.hero { 
    background: linear-gradient(135deg, rgba(255,107,107,0.1) 0%, rgba(255,255,255,0.05) 100%);
    border-radius: 24px;
    padding: 3rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    animation: float 6s ease-in-out infinite;
}
.hero h1 { 
    background: linear-gradient(135deg, #e63946 0%, #ff6b6b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    font-weight: 800;
}
.hero p { 
    color: #adb5bd;
    font-size: 1.2rem;
}

/* Buttons */
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
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 12px 32px rgba(230,57,70,0.45);
}

/* Image Effects */
img {
    border-radius: 16px;
    box-shadow: var(--shadow-medium);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
img:hover {
    transform: scale(1.02);
    box-shadow: var(--shadow-heavy);
}

/* Upload section */
.upload-section {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 2rem;
    border: 2px dashed rgba(255,255,255,0.15);
    text-align: center;
    transition: 0.3s ease;
}
.upload-section:hover {
    border-color: #ff6b6b;
    box-shadow: 0 12px 48px rgba(230,57,70,0.3);
}

/* Animations */
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}
</style>
""", unsafe_allow_html=True)

# =========================
# Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        classes=1,
        activation=None
    )
    model.load_state_dict(torch.load("best_deeplab.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

with st.spinner("🔍 Loading model..."):
    model = load_model()
st.success("✅ Model loaded successfully (best_deeplab.pth)")

# =========================
# Preprocessing / TTA / Postprocess / Extract
# =========================
def preprocess_image(img: Image.Image, long_side=512, pad_to_divisible=16):
    img_np = np.array(img.convert("RGB"))
    h, w = img_np.shape[:2]
    scale = long_side / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    pad_h = (pad_to_divisible - new_h % pad_to_divisible) % pad_to_divisible
    pad_w = (pad_to_divisible - new_w % pad_to_divisible) % pad_to_divisible
    resized = cv2.resize(img_np, (new_w, new_h))
    padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor, (h, w), padded

def tta_predict(img_tensor: torch.Tensor):
    img_tensor = img_tensor.to(device)
    preds = []

    # Original
    preds.append(model(img_tensor))

    # Horizontal flip
    h_flip = torch.flip(img_tensor, dims=[-1])
    preds.append(torch.flip(model(h_flip), dims=[-1]))

    # Vertical flip
    v_flip = torch.flip(img_tensor, dims=[-2])
    preds.append(torch.flip(model(v_flip), dims=[-2]))

    # Rotate 90
    rot = torch.rot90(img_tensor, k=1, dims=[-2, -1])
    preds.append(torch.rot90(model(rot), k=-1, dims=[-2, -1]))

    # Average
    pred = torch.stack(preds, dim=0).mean(dim=0)
    return pred

def postprocess_mask(mask_tensor, orig_size, blur_strength=5):
    mask = mask_tensor.squeeze().detach().cpu().numpy()
    mask = cv2.resize(mask, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0.5).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (blur_strength, blur_strength), 0)
    _, smooth_mask = cv2.threshold(blurred, 0.5, 1, cv2.THRESH_BINARY)
    return smooth_mask.astype(np.uint8)

def extract_object(img, mask, bg_color=(0,0,0), transparent=False, custom_bg=None, gradient=None, bg_blur=False, blur_amount=21):
    mask = mask.astype(np.float32)
    mask = cv2.GaussianBlur(mask, (7,7), 0)[..., np.newaxis]
    mask /= (mask.max() + 1e-8)
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
                alpha = i / max(h-1,1)
                bg_resized[i,:,:] = (1-alpha)*col1 + alpha*col2
        elif custom_bg is not None:
            bg_resized = cv2.resize(np.array(custom_bg), (img.shape[1], img.shape[0])).astype(np.float32)
        else:
            bg_resized = np.full_like(img, bg_color, dtype=np.float32)
        result = img.astype(np.float32)*mask + bg_resized*(1-mask)
        return result.astype(np.uint8)

def create_edge_overlay(image, mask, edge_color="#00FF00", edge_thickness=2):
    edge_color_rgb = tuple(int(edge_color.lstrip('#')[i:i+2],16) for i in (0,2,4))
    pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    mask_resized = Image.fromarray((mask*255).astype(np.uint8)) if isinstance(mask, np.ndarray) else mask
    if mask_resized.size != pil_image.size:
        mask_resized = mask_resized.resize(pil_image.size, Image.NEAREST)
    mask_array = np.array(mask_resized)
    contours = measure.find_contours(mask_array, 128)
    overlay_edges = pil_image.copy()
    draw = ImageDraw.Draw(overlay_edges)
    for contour in contours:
        contour_scaled = contour * (pil_image.size[0]/mask_resized.width)
        contour_points = [tuple(p[::-1]) for p in contour_scaled]
        if len(contour_points)>1:
            draw.line(contour_points, fill=edge_color_rgb, width=edge_thickness)
    return np.array(overlay_edges)

# =========================
# Hero Section
# =========================
st.markdown("""
<div class='hero'>
    <h1>✨ AI Object Segmentation</h1>
    <p>Transform your images with precision AI-powered object extraction</p>
</div>
""", unsafe_allow_html=True)

# =========================
# Sidebar Options
# =========================
st.sidebar.markdown("### ⚙️ Segmentation Settings")
st.sidebar.markdown("---")
with st.sidebar.expander("🎨 Quality Settings", expanded=True):
    tta_toggle = st.checkbox("🔄 High Quality Mode (TTA)", True)
    blur_strength = st.slider("✨ Edge Smoothness", 3, 21, 7, step=2)
    dilate_iter = st.slider("📏 Mask Expansion", 0, 5, 1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎭 Background Settings")
use_transparent = st.sidebar.checkbox("🔲 Transparent Background")
bg_type = st.sidebar.radio("Background Style", ["Solid Color","Gradient","Custom Image","Blur Background"])

gradient_colors = None
bg_blur = False
blur_amount = 21
custom_bg = None

if bg_type == "Solid Color":
    bg_color = st.sidebar.color_picker("🎨 Background Color", "#000000")
    bg_tuple = tuple(int(bg_color.lstrip('#')[i:i+2],16) for i in (0,2,4))
elif bg_type == "Custom Image":
    bg_file = st.sidebar.file_uploader("📸 Upload Background Image", type=["jpg","png"])
    custom_bg = Image.open(bg_file).convert("RGB") if bg_file else None
    bg_tuple = (0,0,0)
elif bg_type == "Gradient":
    col1_exp = st.sidebar.columns(2)
    with col1_exp[0]: color1 = st.color_picker("Start", "#000000")
    with col1_exp[1]: color2 = st.color_picker("End", "#ff0000")
    col1 = np.array([int(color1.lstrip('#')[i:i+2],16) for i in (0,2,4)], dtype=np.uint8)
    col2 = np.array([int(color2.lstrip('#')[i:i+2],16) for i in (0,2,4)], dtype=np.uint8)
    gradient_colors = (col1, col2)
    bg_tuple = (0,0,0)
elif bg_type == "Blur Background":
    bg_blur = True
    blur_amount = st.sidebar.slider("🌫️ Blur Intensity", 5, 99, 21, step=2)
    bg_tuple = (0,0,0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Artistic Effects")
with st.sidebar.expander("🖌️ Apply Artistic Style", expanded=False):
    artistic_effect = st.selectbox(
        "Effect Type", ["None", "Sketch", "Cartoon", "Oil Painting", "Watercolor", "Colored Pencil"]
    )
    effect_intensity = st.slider("🎚️ Effect Intensity", 0.0, 1.0, 0.8, 0.1) if artistic_effect != "None" else 1.0

st.sidebar.markdown("---")
st.sidebar.markdown("### 🖍️ Edge Overlay Settings")
with st.sidebar.expander("✏️ Edge Visualization", expanded=False):
    show_edges = st.checkbox("Show Edge Overlay", value=False)
    if show_edges:
        edge_color = st.color_picker("Edge Color", "#00FF00")
        edge_thickness = st.slider("Edge Thickness", 1, 10, 2)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** High Quality Mode works best for complex objects!")

# =========================
# Mode Selection
# =========================
st.markdown("### 🖼️ Select Processing Mode")
mode = st.radio("Select Processing Mode", ["Single Image", "Batch Processing"], horizontal=True, label_visibility="collapsed")


# =========================
# Single Image Processing
# =========================
if mode == "Single Image":
    uploaded_file = st.file_uploader("📤 Upload Your Image", type=["jpg","jpeg","png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        st.image(img, caption="📷 Uploaded Image", use_container_width=True)

        if st.button("✨ Process Image", use_container_width=True):
            img_tensor, orig_size, _ = preprocess_image(img)
            mask_tensor = tta_predict(img_tensor) if tta_toggle else model(img_tensor)
            pred_mask = postprocess_mask(mask_tensor, orig_size, blur_strength=blur_strength)
            for _ in range(dilate_iter):
                pred_mask = cv2.dilate(pred_mask, np.ones((3,3), np.uint8), iterations=1)

            result = extract_object(
                img_np, pred_mask,
                bg_color=bg_tuple,
                transparent=use_transparent,
                custom_bg=custom_bg,
                gradient=gradient_colors,
                bg_blur=bg_blur,
                blur_amount=blur_amount
            )

            edge_overlay = create_edge_overlay(img_np, pred_mask, edge_color=edge_color, edge_thickness=edge_thickness) if show_edges else None

            st.image(result, caption="✨ Segmented Image", use_container_width=True)
            if show_edges and edge_overlay is not None:
                st.image(edge_overlay, caption="✏️ Edge Overlay", use_container_width=True)

# =========================
# Batch Processing
# =========================
elif mode == "Batch Processing":
    uploaded_files = st.file_uploader("📤 Upload Multiple Images", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded_files:
        images = [np.array(Image.open(f).convert("RGB")) for f in uploaded_files]
        st.markdown(f"### 📸 Uploaded {len(images)} images")
        if st.button("✨ Process All Images"):
            results, edge_overlays = [], []
            progress = st.progress(0, text="🚀 Starting batch processing...")
            for i, img_np in enumerate(images):
                img_tensor, orig_size, _ = preprocess_image(Image.fromarray(img_np))
                mask_tensor = tta_predict(img_tensor) if tta_toggle else model(img_tensor)
                pred_mask = postprocess_mask(mask_tensor, orig_size, blur_strength=blur_strength)
                for _ in range(dilate_iter):
                    pred_mask = cv2.dilate(pred_mask, np.ones((3,3), np.uint8), iterations=1)
                result = extract_object(
                    img_np, pred_mask,
                    bg_color=bg_tuple,
                    transparent=use_transparent,
                    custom_bg=custom_bg,
                    gradient=gradient_colors,
                    bg_blur=bg_blur,
                    blur_amount=blur_amount
                )
                results.append(result)
                if show_edges:
                    edge_overlay = create_edge_overlay(img_np, pred_mask, edge_color=edge_color, edge_thickness=edge_thickness)
                    edge_overlays.append(edge_overlay)
                progress.progress(int((i+1)/len(images)*100), text=f"Processing image {i+1}/{len(images)}")

            st.success(f"✅ Successfully processed {len(results)} images!")

            # Display first few results
            cols = st.columns(min(len(results),4))
            for idx, (col, res) in enumerate(zip(cols, results[:4])):
                col.image(res, caption=f"Result {idx+1}", use_container_width=True)

            # ZIP download
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer,"w") as zip_file:
                for idx, res in enumerate(results):
                    buf = BytesIO()
                    Image.fromarray(res).save(buf, format="PNG")
                    zip_file.writestr(f"segmented_{idx+1}.png", buf.getvalue())
               