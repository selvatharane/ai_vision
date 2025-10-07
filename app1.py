# app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image, ImageDraw
import torchvision.transforms as T
import numpy as np
import cv2
from io import BytesIO
import os
import time
from skimage import measure
import zipfile
from streamlit_image_comparison import image_comparison

# ===========================
# Page Setup (your original UI + CSS)
# ===========================
st.set_page_config(
    page_title="AI Vision Extraction", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.sidebar.title("📘 About This App")
st.sidebar.info(
    """
    AI Vision Extraction allows you to automatically isolate the main object from any image 
    using an advanced **DeepLabV3** segmentation model with TTA and enhanced post-processing.
    """
)
st.sidebar.markdown("🎯 **How to Use**")
st.sidebar.markdown("""
1. Upload an image (JPG, JPEG, or PNG).  
2. The AI extracts the main object and removes the background.  
3. Preview & download your extracted image instantly.  
""")

# Your custom CSS (kept from your second snippet)
st.markdown("""
<style>
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.stApp {
    background: linear-gradient(270deg, #ff9a9e, #fad0c4, #a1c4fd, #c2e9fb);
    background-size: 800% 800%;
    animation: gradientBG 15s ease infinite;
    color: #000000;
}
[data-testid="stSidebar"] {
    background-color: #f0f2f6;
    color: #000000;
    font-weight: 500;
}
button[title="Toggle sidebar"] {
    color: #000000 !important;
    background-color: #e0e0e0 !important;
    border-radius: 5px;
    z-index: 9999;
    transition: all 0.3s ease;
}
button[title="Toggle sidebar"]:hover {
    background-color: #c0c0c0 !important;
    transform: scale(1.05);
}
.center-img {
    display: flex;
    justify-content: center;
    margin: 20px 0;
}
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-track { background: #f0f2f6; }
::-webkit-scrollbar-thumb { background: #c0c0c0; border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: #888888; }
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    border-radius: 5px;
    transition: transform 0.2s, background-color 0.3s;
}
.stButton>button:hover {
    background-color: #45a049;
    transform: scale(1.05);
}
.css-1v3fvcr, .css-10trblm { color: #000000 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 AI Vision Extraction App")
st.caption("Upload an image to extract the object using DeepLabV3 segmentation")

sample_img = "sample_extraction.jpg"
if os.path.exists(sample_img):
    st.markdown('<div class="center-img">', unsafe_allow_html=True)
    st.image(sample_img, caption="Example: Object Extraction", use_container_width=False, width=400)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("⚠️ Add a 'sample_extraction.jpg' in your repo to show a preview here.")

# ===========================
# Sidebar — options merged from your first code
# ===========================
st.sidebar.markdown("### ⚙️ Segmentation Settings")
tta_toggle = st.sidebar.checkbox("🔄 High Quality Mode (TTA)", True)
blur_strength = st.sidebar.slider("✨ Edge Smoothness", 3, 21, 7, step=2)
dilate_iter = st.sidebar.slider("📏 Mask Expansion", 0, 5, 1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎭 Background Settings")
use_transparent = st.sidebar.checkbox("🔲 Transparent Background", help="Create PNG with transparency")

bg_type = st.sidebar.radio("Background Style", ["Solid Color","Gradient","Custom Image","Blur Background"])
gradient_colors = None
bg_blur = False
blur_amount = 21
custom_bg = None
bg_tuple = (0,0,0)

if bg_type == "Solid Color":
    bg_color = st.sidebar.color_picker("🎨 Background Color", "#000000")
    bg_tuple = tuple(int(bg_color.lstrip('#')[i:i+2],16) for i in (0,2,4))
elif bg_type == "Custom Image":
    bg_file = st.sidebar.file_uploader("📸 Upload Background Image", type=["jpg","png"])
    custom_bg = Image.open(bg_file).convert("RGB") if bg_file else None
elif bg_type == "Gradient":
    col1_exp = st.sidebar.columns(2)
    with col1_exp[0]:
        color1 = st.color_picker("Start", "#000000")
    with col1_exp[1]:
        color2 = st.color_picker("End", "#ff0000")
    col1 = np.array([int(color1.lstrip('#')[i:i+2], 16) for i in (0,2,4)], dtype=np.uint8)
    col2 = np.array([int(color2.lstrip('#')[i:i+2], 16) for i in (0,2,4)], dtype=np.uint8)
    gradient_colors = (col1, col2)
elif bg_type == "Blur Background":
    bg_blur = True
    blur_amount = st.sidebar.slider("🌫️ Blur Intensity", 5, 99, 21, step=2)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Artistic Effects")
artistic_effect = st.sidebar.selectbox("Effect Type", ["None", "Sketch", "Cartoon", "Oil Painting", "Watercolor", "Colored Pencil"])
effect_intensity = 1.0
if artistic_effect != "None":
    effect_intensity = st.sidebar.slider("🎚️ Effect Intensity", 0.0, 1.0, 0.8, 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🖍️ Edge Overlay Settings")
show_edges = st.sidebar.checkbox("Show Edge Overlay", value=False)
edge_color = "#00FF00"
edge_thickness = 2
if show_edges:
    edge_color = st.sidebar.color_picker("Edge Color", "#00FF00")
    edge_thickness = st.sidebar.slider("Edge Thickness", 1, 10, 2)

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: TTA improves results for complex objects.")

# ===========================
# Device & Model Load (replaced with your deeplabv3-based loader)
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model(model_path="best_deeplab.pth", num_classes=2):
    # Build torchvision deeplabv3_resnet50 and adjust classifier to match classes
    model = deeplabv3_resnet50(weights=None, aux_loss=True)
    # Modify classifier if necessary to output num_classes
    try:
        old_cls = model.classifier
        # Build a small head keeping existing blocks if shape matches
        model.classifier = nn.Sequential(
            old_cls[0], old_cls[1], old_cls[2], nn.Dropout(0.3),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
    except Exception:
        # fallback: try to replace last conv if previous modification schema doesn't fit
        try:
            model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
        except Exception:
            pass

    # load weights (user-provided)
    checkpoint = torch.load(model_path, map_location=device)
    # load_state_dict flexible load
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model

# attempt to load model
try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# ===========================
# Artistic Effects functions (copied & adapted)
# ===========================
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
    for _ in range(num_down): img_color = cv2.pyrDown(img_color)
    for _ in range(num_bilateral): img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
    for _ in range(num_down): img_color = cv2.pyrUp(img_color)
    img_color = cv2.resize(img_color, (img.shape[1], img.shape[0]))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
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

def apply_artistic_effect(img, effect_type, intensity=1.0):
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

# ===========================
# Edge overlay (copied)
# ===========================
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

# ===========================
# Preprocess / TTA / Postprocess (from your first code)
# ===========================
def preprocess_image_for_model(img_pil, long_side=256):
    img = np.array(img_pil)
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    new_w, new_h = int(w*scale), int(h*scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    img_tensor = torch.tensor(img_resized/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor.to(device), (h, w), img

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
            out = model(t)['out'] if isinstance(model(t), dict) else model(t)
            if isinstance(out, tuple): out = out[0]
            # Ensure out is logits with shape (B, C, H, W)
            if f != aug_list[0]:
                if f == aug_list[1]: out = torch.flip(out,[3])
                elif f == aug_list[2]: out = torch.flip(out,[2])
                elif f == aug_list[3]: out = torch.rot90(out,k=3,dims=[2,3])
                elif f == aug_list[4]: out = torch.rot90(out,k=1,dims=[2,3])
            outputs.append(out)
    return torch.mean(torch.stack(outputs), dim=0)

def postprocess_mask(mask_tensor, orig_size, blur_strength=7):
    # mask_tensor expected from model: logits (B, C, H, W)
    if mask_tensor.ndim == 4:
        mask = torch.argmax(mask_tensor, dim=1).squeeze().cpu().numpy()
    else:
        # if single channel probability map
        mask = mask_tensor.squeeze().cpu().numpy()
        # convert probability -> binary
        mask = (mask > 0.5).astype(np.uint8)
    mask = cv2.resize(mask.astype(np.uint8), (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = mask.astype(np.uint8) * 255
    blurred = cv2.GaussianBlur(mask, (blur_strength, blur_strength), 0)
    _, smooth_mask = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
    return (smooth_mask // 255).astype(np.uint8)

# ===========================
# Extraction helper (kept advanced with transparent/gradient/custom bg)
# ===========================
def extract_object_with_bg(img, mask, bg_color=(0,0,0), transparent=False, custom_bg=None, gradient=None, bg_blur=False, blur_amount=21):
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

# ===========================
# Mode selection & UI flow
# ===========================
st.header("🖼️ Try Your Own Image")
mode = st.radio("Select Processing Mode", ["Single Image", "Batch Processing"], horizontal=True, label_visibility="collapsed")

if mode == "Single Image":
    uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg","jpeg","png"])
    if uploaded_file is not None and model is not None:
        input_img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(input_img)

        col1, col2 = st.columns([1,1])
        with col1:
            st.image(input_img, caption="Original Image", use_container_width=True)

        process_btn = st.button("✨ Process Image")
        if process_btn:
            progress = st.progress(0, text="🔄 Initializing...")
            with st.spinner("🎨 AI is working..."):
                # progress simulation with updates
                for i, txt in enumerate(["Analyzing image...", "Detecting objects...", "Segmenting...", "Refining edges...", "Finalizing..."]):
                    progress.progress(i*20, text=f"🔍 {txt}")
                    time.sleep(0.15)

                # Preprocess & predict
                img_tensor, orig_size, _ = preprocess_image_for_model(input_img)
                mask_tensor = tta_predict(img_tensor) if tta_toggle else (model(img_tensor)['out'] if isinstance(model(img_tensor), dict) else model(img_tensor))
                pred_mask = postprocess_mask(mask_tensor, orig_size, blur_strength=blur_strength)
                for _ in range(dilate_iter):
                    pred_mask = cv2.dilate(pred_mask, np.ones((3,3),np.uint8), iterations=1)

                # Apply artistic effect to original image copy
                if artistic_effect != "None":
                    img_np_styled = apply_artistic_effect(img_np, artistic_effect, effect_intensity)
                else:
                    img_np_styled = img_np

                result = extract_object_with_bg(img_np_styled, pred_mask, bg_color=bg_tuple, transparent=use_transparent, custom_bg=custom_bg, gradient=gradient_colors, bg_blur=bg_blur, blur_amount=blur_amount)
                edge_overlay = None
                if show_edges:
                    edge_overlay = create_edge_overlay(img_np, pred_mask, edge_color=edge_color, edge_thickness=edge_thickness)

            progress.empty()
            st.success("✅ Segmentation Complete!")

            # Tabs & outputs
            if show_edges:
                tabs = st.tabs(["🖼️ Side by Side", "↔️ Interactive Comparison", "✏️ Edge Overlay"])
            else:
                tabs = st.tabs(["🖼️ Side by Side", "↔️ Interactive Comparison"])

            with tabs[0]:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**📷 Original**")
                    st.image(img_np, use_container_width=True)
                with c2:
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
                result_bg = extract_object_with_bg(img_np_styled, pred_mask, bg_color=bg_tuple, transparent=False, custom_bg=custom_bg, gradient=gradient_colors, bg_blur=bg_blur, blur_amount=blur_amount)
                overlay_small = cv2.resize(result_bg, (500, int(500*result_bg.shape[0]/result_bg.shape[1])))
                image_comparison(img1=img_small, img2=overlay_small, label1="Original", label2="Segmented")

            if show_edges:
                with tabs[2]:
                    st.markdown("**✏️ Edge Boundaries Overlay**")
                    st.image(edge_overlay, caption="Edges Overlay", use_container_width=True)
                    st.markdown("---")
                    buf_edge = BytesIO()
                    Image.fromarray(edge_overlay).save(buf_edge, format="PNG")
                    st.download_button("💾 Download Edge Overlay", buf_edge.getvalue(), "edge_overlay.png", "image/png", use_container_width=True)

elif mode == "Batch Processing":
    uploaded_files = st.file_uploader("📤 Upload Multiple Images", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded_files and model is not None:
        st.markdown(f"### 📸 Uploaded {len(uploaded_files)} images")
        images = [np.array(Image.open(f).convert("RGB")) for f in uploaded_files]
        cols = st.columns(min(len(images), 4))
        for idx, (col, img_np) in enumerate(zip(cols, images[:4])):
            col.image(img_np, caption=f"Image {idx+1}", use_container_width=True)
        if len(images) > 4:
            st.info(f"📋 {len(images)-4} more images ready for processing...")

        process_all = st.button("✨ Process All Images")
        if process_all:
            results = []
            edge_overlays = []
            progress = st.progress(0, text="🚀 Starting batch processing...")
            for i, img_np in enumerate(images):
                progress.progress(int((i)/len(images)*100), text=f"🎨 Processing image {i+1} of {len(images)}...")
                pil = Image.fromarray(img_np)
                img_tensor, orig_size, _ = preprocess_image_for_model(pil)
                mask_tensor = tta_predict(img_tensor) if tta_toggle else (model(img_tensor)['out'] if isinstance(model(img_tensor), dict) else model(img_tensor))
                pred_mask = postprocess_mask(mask_tensor, orig_size, blur_strength=blur_strength)
                for _ in range(dilate_iter):
                    pred_mask = cv2.dilate(pred_mask, np.ones((3,3),np.uint8), iterations=1)

                if artistic_effect != "None":
                    img_np_styled = apply_artistic_effect(img_np, artistic_effect, effect_intensity)
                else:
                    img_np_styled = img_np

                result = extract_object_with_bg(img_np_styled, pred_mask, bg_color=bg_tuple, transparent=use_transparent, custom_bg=custom_bg, gradient=gradient_colors, bg_blur=bg_blur, blur_amount=blur_amount)
                results.append(result)
                if show_edges:
                    edge_overlay = create_edge_overlay(img_np, pred_mask, edge_color=edge_color, edge_thickness=edge_thickness)
                    edge_overlays.append(edge_overlay)

            progress.progress(100, text="✅ All images processed!")
            time.sleep(0.3)
            progress.empty()
            st.success(f"✅ Successfully processed {len(results)} images!")
            st.markdown("---")
            cols = st.columns(min(len(results), 4))
            for idx, (col, result) in enumerate(zip(cols, results[:4])):
                col.image(result, caption=f"Result {idx+1}", use_container_width=True)

            if len(results) > 4:
                with st.expander(f"👁️ View all {len(results)} results"):
                    remaining_cols = st.columns(4)
                    for idx, result in enumerate(results[4:], 5):
                        remaining_cols[(idx-5) % 4].image(result, caption=f"Result {idx}", use_container_width=True)

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

            # Zip download
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for idx, result in enumerate(results):
                    buf = BytesIO()
                    Image.fromarray(result).save(buf, format="PNG")
                    zip_file.writestr(f"segmented_{idx+1}.png", buf.getvalue())
                if show_edges and edge_overlays:
                    for idx, edge_img in enumerate(edge_overlays):
                        buf = BytesIO()
                        Image.fromarray(edge_img).save(buf, format="PNG")
                        zip_file.writestr(f"edge_overlay_{idx+1}.png", buf.getvalue())

            col_dl1, col_dl2, col_dl3 = st.columns([1,1,1])
            with col_dl2:
                st.download_button("📦 Download All as ZIP", zip_buffer.getvalue(), "segmented_images.zip", "application/zip", use_container_width=True)
