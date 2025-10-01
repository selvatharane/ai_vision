import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import cv2
import os
from io import BytesIO

# ===========================
# 1. Page Title
# ===========================
st.title("AI Vision App")
st.subheader("Sample Vision Extraction")

# ===========================
# 2. Show Example Image
# ===========================
sample_img = "sample_extraction.jpg"  # keep this in repo root or images/ folder
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
def preprocess_image(image: Image.Image):
    img = image.convert("RGB")
    img = np.array(img)
    img_resized = cv2.resize(img, (256, 256))  # resize to model input
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return img_tensor

def predict_mask(model, device, image: Image.Image):
    img_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)
    
    # Convert mask to color
    mask_color = cv2.applyColorMap(mask*255, cv2.COLORMAP_JET)
    mask_color = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)
    return mask_color

# Helper to convert image to bytes for download
def get_image_download_link(img_array, filename="mask.png"):
    pil_img = Image.fromarray(img_array)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

# ===========================
# 5. Upload & Test
# ===========================
st.header("Try Your Own Image")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    input_img = Image.open(uploaded_file)

    # Get prediction
    mask_color = predict_mask(model, device, input_img)

    # Show results side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(input_img, caption="Original Image", use_container_width=True)
    with col2:
        st.image(mask_color, caption="Predicted Mask", use_container_width=True)

    # Download button below
    mask_bytes = get_image_download_link(mask_color)
    st.download_button(
        label="Download Mask",
        data=mask_bytes,
        file_name="predicted_mask.png",
        mime="image/png"
    )
