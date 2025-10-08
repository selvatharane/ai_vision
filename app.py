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
# Artistic Effect Functions (fixed)
# =========================
def apply_sketch_effect(img, intensity=1.0):
    img = img.astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inv_gray = 255 - gray
    blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256.0)
    sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(img, 1 - intensity, sketch_rgb, intensity, 0).astype(np.uint8)

def apply_cartoon_effect(img, intensity=1.0):
    img = img.astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    num_down, num_bilateral = 2, 7
    img_color = bgr.copy()
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)
    img_color = cv2.resize(img_color, (img.shape[1], img.shape[0]))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 2)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(img_color, edges)
    cartoon_rgb = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 1 - intensity, cartoon_rgb, intensity, 0).astype(np.uint8)

def apply_oil_paint_effect(img, intensity=1.0):
    img = img.astype(np.uint8)
    result = img.copy()
    iterations = int(3 + 2 * intensity)
    for _ in range(iterations):
        result = cv2.bilateralFilter(result, 9, 75, 75)
    result = cv2.medianBlur(result, 5)
    hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return cv2.addWeighted(img, 1 - intensity, result, intensity, 0).astype(np.uint8)

def apply_watercolor_effect(img, intensity=1.0):
    img = img.astype(np.uint8)
    try:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        stylized = cv2.stylization(bgr, sigma_s=60, sigma_r=0.6)
        stylized_rgb = cv2.cvtColor(stylized, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(img, 1 - intensity, stylized_rgb, intensity, 0).astype(np.uint8)
    except:
        return img

def apply_pencil_color_effect(img, intensity=1.0):
    img = img.astype(np.uint8)
    try:
        _, pencil_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        return cv2.addWeighted(img, 1 - intensity, pencil_color, intensity, 0).astype(np.uint8)
    except:
        return img

def apply_artistic_effect(img, effect_type, intensity=1.0):
    if effect_type == "None":
        return img
    if effect_type == "Sketch": return apply_sketch_effect(img, intensity)
    if effect_type == "Cartoon": return apply_cartoon_effect(img, intensity)
    if effect_type == "Oil Painting": return apply_oil_paint_effect(img, intensity)
    if effect_type == "Watercolor": return apply_watercolor_effect(img, intensity)
    if effect_type == "Colored Pencil": return apply_pencil_color_effect(img, intensity)
    return img

# =========================
# Preprocessing / TTA / Postprocess
# =========================
def preprocess_image(img: Image.Image, long_side=512, pad_to_divisible=16):
    img = np.array(img.convert("RGB"))
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    pad_h = (pad_to_divisible - new_h % pad_to_divisible) % pad_to_divisible
    pad_w = (pad_to_divisible - new_w % pad_to_divisible) % pad_to_divisible
    resized = cv2.resize(img, (new_w, new_h))
    padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    tensor = torch.from_numpy(padded).permute(2,0,1).unsqueeze(0).float()/255.0
    return tensor, (h,w), padded

def tta_predict(img_tensor: torch.Tensor):
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device, dtype=torch.float32)
    preds = []
    preds.append(model(img_tensor)['out'])
    h_flip = torch.flip(img_tensor, dims=[-1])
    preds.append(torch.flip(model(h_flip)['out'], dims=[-1]))
    v_flip = torch.flip(img_tensor, dims=[-2])
    preds.append(torch.flip(model(v_flip)['out'], dims=[-2]))
    rot = torch.rot90(img_tensor, k=1, dims=[-2,-1])
    preds.append(torch.rot90(model(rot)['out'], k=-1, dims=[-2,-1]))
    return torch.stack(preds).mean(dim=0)

def postprocess_mask(mask_tensor, orig_size, blur_strength=5):
    mask = torch.argmax(mask_tensor, dim=1).squeeze().cpu().numpy()
    mask = cv2.resize(mask.astype(np.uint8), (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = mask.astype(np.uint8) * 255
    blurred = cv2.GaussianBlur(mask, (blur_strength, blur_strength),0)
    _, smooth_mask = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
    return (smooth_mask//255).astype(np.uint8)

# =========================
# Edge Overlay
# =========================
def create_edge_overlay(image, mask, edge_color="#00FF00", edge_thickness=2):
    edge_color_rgb = tuple(int(edge_color.lstrip('#')[i:i+2],16) for i in (0,2,4))
    pil_image = Image.fromarray(image) if isinstance(image,np.ndarray) else image
    mask_resized = Image.fromarray((mask*255).astype(np.uint8)) if isinstance(mask,np.ndarray) else mask
    if mask_resized.size != pil_image.size:
        mask_resized = mask_resized.resize(pil_image.size, Image.NEAREST)
    mask_array = np.array(mask_resized)
    contours = measure.find_contours(mask_array, 128)
    overlay_edges = pil_image.copy()
    draw = ImageDraw.Draw(overlay_edges)
    for contour in contours:
        contour_scaled = contour*(pil_image.size[0]/mask_resized.width)
        contour_points = [tuple(p[::-1]) for p in contour_scaled]
        if len(contour_points)>1:
            draw.line(contour_points, fill=edge_color_rgb, width=edge_thickness)
    return np.array(overlay_edges)

# =========================
# Object Extraction
# =========================
def extract_object(img, mask, bg_color=(0,0,0), transparent=False, custom_bg=None, gradient=None, bg_blur=False, blur_amount=21):
    mask = mask.astype(np.float32)
    mask = cv2.GaussianBlur(mask,(7,7),0)
    mask = np.expand_dims(mask,-1)
    mask = mask/(mask.max()+1e-8)
    img_float = img.astype(np.float32)
    fg = img_float*mask
    if transparent:
        bg = np.zeros_like(img_float)
        result = np.concatenate([fg,bg[:,:,0:1]],axis=-1).astype(np.uint8)
    elif custom_bg is not None:
        bg = np.array(custom_bg).astype(np.float32)
        bg = cv2.resize(bg, (img.shape[1], img.shape[0]))
        result = (fg + bg*(1-mask)).astype(np.uint8)
    elif gradient is not None:
        grad = np.linspace(gradient[0], gradient[1], img.shape[0], dtype=np.uint8)
        grad = np.tile(grad[:,None,:], (1,img.shape[1],1))
        result = (fg + grad*(1-mask)).astype(np.uint8)
    elif bg_blur:
        blur_bg = cv2.GaussianBlur(img,(blur_amount,blur_amount),0)
        result = (fg + blur_bg*(1-mask)).astype(np.uint8)
    else:
        bg = np.ones_like(img_float)*np.array(bg_color,dtype=np.float32)
        result = (fg + bg*(1-mask)).astype(np.uint8)
    return result.astype(np.uint8)


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
# SINGLE IMAGE
# =========================
if mode == "Single Image":
    st.markdown("---")
    uploaded_file = st.file_uploader("📤 Upload Your Image", type=["jpg","jpeg","png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        st.image(img, caption="📷 Uploaded Image", width=400)

        process_btn = st.button("✨ Process Image")
        if process_btn:
            progress = st.progress(0, text="🔄 Initializing AI model...")
            msgs = ["🔄 Initializing AI model...", "🔍 Analyzing image...", "🎯 Detecting objects...",
                    "✂️ Segmenting...", "🎨 Refining edges...", "✅ Finalizing..."]
            for i in range(0, 101, 20):
                idx = min(i // 20, len(msgs)-1)
                progress.progress(i, text=msgs[idx])
                time.sleep(0.2)

            # Preprocess and predict
            img_tensor, orig_size, _ = preprocess_image(img, long_side=512, pad_to_divisible=16)
            mask_tensor = tta_predict(img_tensor) if tta_toggle else model(img_tensor)['out']

            # Postprocess mask
            pred_mask = postprocess_mask(mask_tensor, orig_size, blur_strength=blur_strength)
            for _ in range(dilate_iter):
                pred_mask = cv2.dilate(pred_mask, np.ones((3,3), np.uint8), iterations=1)

            # Extract object
            result = extract_object(
                img_np, pred_mask,
                bg_color=bg_tuple,
                transparent=use_transparent,
                custom_bg=custom_bg,
                gradient=gradient_colors,
                bg_blur=bg_blur,
                blur_amount=blur_amount
            )

            # Apply artistic effect safely
            if artistic_effect != "None":
                if result.shape[2] == 4:
                    alpha = result[:, :, 3]
                    result_rgb = result[:, :, :3]  # RGBA -> RGB slice
                    result_rgb = apply_artistic_effect(result_rgb, artistic_effect, effect_intensity)
                    result = np.dstack([result_rgb, alpha])
                else:
                    result = apply_artistic_effect(result, artistic_effect, effect_intensity)

            # Edge overlay
            edge_overlay = create_edge_overlay(img_np, pred_mask, edge_color=edge_color, edge_thickness=edge_thickness) if show_edges else None

            progress.progress(100, text="✅ Segmentation Complete!")
            st.success("✅ Segmentation Complete!")
            st.markdown("---")

            # Tabs
            tabs = st.tabs(["🖼️ Side by Side", "↔️ Interactive Comparison"] + (["✏️ Edge Overlay"] if show_edges else []))

            with tabs[0]:
                col1, col2 = st.columns(2)
                with col1: st.image(img_np, caption="📷 Original", width=400)
                with col2: st.image(result, caption="✨ Segmented", width=400)

            with tabs[1]:
                from streamlit_image_comparison import image_comparison
                image_comparison(
                    img1=img_np,
                    img2=result if result.shape[2]==3 else result[:,:,:3],
                    label1="Original",
                    label2="Processed",
                    width=700
                )

            if show_edges and edge_overlay is not None:
                with tabs[2]:
                    st.image(edge_overlay, caption="✏️ Edge Overlay", width=400)

            # Download buttons
            buf_seg = BytesIO()
            Image.fromarray(result).save(buf_seg, format="PNG")
            st.download_button("💾 Download Segmented", buf_seg.getvalue(), "segmented_image.png", "image/png")
            
            if show_edges and edge_overlay is not None:
                buf_edge = BytesIO()
                Image.fromarray(edge_overlay).save(buf_edge, format="PNG")
                st.download_button("💾 Download Edge Overlay", buf_edge.getvalue(), "edge_overlay.png", "image/png")

# =========================
# BATCH PROCESSING
# =========================
elif mode == "Batch Processing":
    st.markdown("---")
    uploaded_files = st.file_uploader("📤 Upload Multiple Images", type=["jpg","jpeg","png"], accept_multiple_files=True)
    
    if uploaded_files:
        images = [np.array(Image.open(f).convert("RGB")).astype(np.uint8) for f in uploaded_files]
        st.markdown(f"### 📸 Uploaded {len(images)} images")

        process_all = st.button("✨ Process All Images")
        if process_all:
            results, edge_overlays = [], []
            progress = st.progress(0, text="🚀 Starting batch processing...")

            msgs = ["🔄 Initializing AI model...", "🔍 Analyzing image...", "🎯 Detecting objects...",
                    "✂️ Segmenting...", "🎨 Refining edges...", "✅ Finalizing..."]

            for i, img_np in enumerate(images):
                for j, p in enumerate(range(0, 101, 20)):
                    idx = min(j, len(msgs)-1)
                    progress.progress(int((i + p/100)/len(images)*100), text=f"Image {i+1}: {msgs[idx]}")
                    time.sleep(0.05)

                # Preprocess and predict
                img_tensor, orig_size, _ = preprocess_image(Image.fromarray(img_np), long_side=512, pad_to_divisible=16)
                mask_tensor = tta_predict(img_tensor) if tta_toggle else model(img_tensor)['out']

                # Postprocess mask
                pred_mask = postprocess_mask(mask_tensor, orig_size, blur_strength=blur_strength)
                for _ in range(dilate_iter):
                    pred_mask = cv2.dilate(pred_mask, np.ones((3,3), np.uint8), iterations=1)

                # Extract object
                result = extract_object(
                    img_np, pred_mask,
                    bg_color=bg_tuple,
                    transparent=use_transparent,
                    custom_bg=custom_bg,
                    gradient=gradient_colors,
                    bg_blur=bg_blur,
                    blur_amount=blur_amount
                )

                # Artistic effect
                if artistic_effect != "None":
                    if result.shape[2] == 4:
                        alpha = result[:, :, 3]
                        result_rgb = result[:, :, :3]
                        result_rgb = apply_artistic_effect(result_rgb, artistic_effect, effect_intensity)
                        result = np.dstack([result_rgb, alpha])
                    else:
                        result = apply_artistic_effect(result, artistic_effect, effect_intensity)

                results.append(result)

                # Edge overlay if enabled
                if show_edges:
                    edge_overlay = create_edge_overlay(img_np, pred_mask, edge_color=edge_color, edge_thickness=edge_thickness)
                    edge_overlays.append(edge_overlay)

            progress.progress(100, text="✅ All images processed!")
            st.success(f"✅ Successfully processed {len(results)} images!")

            # Preview first few results
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
            st.download_button("💾 Download All Images (ZIP)", zip_buffer.getvalue(), "segmented_images.zip")
