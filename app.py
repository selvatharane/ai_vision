import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import cv2
import os
from io import BytesIO

# ===========================
# 1. Title and Description
# ===========================
st.title("🎯 AI Vision - Object Extraction")
st.subheader("Accurate DeepLabV3+ Extraction with Enhanced Post-processing & TTA")

# ===========================
# 2. Show Example
# ===========================
sample_img = "sample_extraction.jpeg"
if os.path.exists(sample_img):
    st.image(sample_img, caption="Example: Input vs Extracted Output", use_container_width=True)
else:
    st.warning("Sample image not found! Please add sample_extraction.jpg to your repo.")

# ===========================
# 3. Load Model
# ===========================
@st.cache_resource
def load_model(model_path="best_deeplab.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

try:
    model, device = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    model, device = None, None

# ===========================
# 4. Helper Functions
# ===========================

# Preprocessing (same as training)
def preprocess_image(image):
    img = image.convert("RGB")
    img = np.array(img)
    img_resized = cv2.resize(img, (256, 256))
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor, img

# --- Test Time Augmentation ---
def predict_with_tta(model, tensor):
    aug_list = [
        lambda x: x,
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(x, [2]),
        lambda x: torch.rot90(x, k=1, dims=[2, 3]),
        lambda x: torch.rot90(x, k=3, dims=[2, 3])
    ]
    outputs = []
    for f in aug_list:
        t = f(tensor)
        with torch.no_grad():
            out = model(t)
            out = torch.sigmoid(out)
        # undo transformations
        if f != aug_list[0]:
            if f == aug_list[1]:
                out = torch.flip(out, [3])
            elif f == aug_list[2]:
                out = torch.flip(out, [2])
            elif f == aug_list[3]:
                out = torch.rot90(out, k=3, dims=[2, 3])
            elif f == aug_list[4]:
                out = torch.rot90(out, k=1, dims=[2, 3])
        outputs.append(out)
    return torch.mean(torch.stack(outputs), dim=0)

# --- Postprocessing (Otsu + Morphology) ---
def postprocess_mask(mask_tensor):
    mask = mask_tensor.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

# --- Extract Object with Background Options ---
def extract_object(original, mask, bg_option, bg_color=(0,0,0)):
    mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]))
    mask_bin = (mask_resized > 128).astype(np.uint8)

    if bg_option == "Blur":
        bg = cv2.GaussianBlur(original, (21,21), 0)
    elif bg_option == "Grayscale":
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        bg = np.stack([gray]*3, axis=-1)
    elif bg_option == "White":
        bg = np.full_like(original, (255,255,255))
    elif bg_option == "Custom Color":
        bg = np.full_like(original, bg_color)
    else:
        bg = np.zeros_like(original)

    return (original * mask_bin[:, :, None] + bg * (1 - mask_bin[:, :, None])).astype(np.uint8)

def overlay_mask(original, mask, alpha=0.5):
    color = np.array([255, 0, 0])  # red overlay
    overlay = original.copy()
    overlay[mask > 128] = (1 - alpha) * overlay[mask > 128] + alpha * color
    return overlay.astype(np.uint8)

# Download helper
def to_bytes(img_array):
    pil_img = Image.fromarray(img_array)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

# ===========================
# 5. Upload & Predict
# ===========================
st.header("Upload Image for Extraction")

uploaded = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded is not None and model is not None:
    image = Image.open(uploaded)

    st.sidebar.header("Customization")
    bg_option = st.sidebar.selectbox("Background", ["Black", "White", "Custom Color", "Blur", "Grayscale"])
    if bg_option == "Custom Color":
        color_hex = st.sidebar.color_picker("Pick Color", "#000000")
        bg_color = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    else:
        bg_color = (0, 0, 0)
    alpha = st.sidebar.slider("Overlay Transparency", 0.0, 1.0, 0.5, 0.05)

    with st.spinner("Running enhanced model with TTA and postprocessing..."):
        tensor, orig = preprocess_image(image)
        tensor = tensor.to(device)
        mask_pred = predict_with_tta(model, tensor)
        mask_np = postprocess_mask(mask_pred)
        extracted = extract_object(orig, mask_np, bg_option, bg_color)
        overlay = overlay_mask(orig, mask_np, alpha)

    # Show results
    st.subheader("Results")
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col1.image(orig, caption="Original", use_container_width=True)
    col2.image(mask_np, caption="Refined Mask", use_container_width=True)
    col3.image(extracted, caption="Extracted Object", use_container_width=True)
    col4.image(overlay, caption="Overlay Preview", use_container_width=True)

    # Downloads
    st.download_button("Download Extracted Object", to_bytes(extracted), "extracted_object.png", "image/png")
    st.download_button("Download Refined Mask", to_bytes(mask_np), "refined_mask.png", "image/png")
