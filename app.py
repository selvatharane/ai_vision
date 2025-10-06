import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import cv2
import base64
import os
from io import BytesIO

# ===========================
# 1. Page Config
# ===========================
st.set_page_config(page_title="AI Vision Extraction", layout="wide")

# ===========================
# 2. Custom Background and Styles
# ===========================
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 50%, #e3f2fd 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a237e 0%, #283593 50%, #303f9f 100%);
    color: #ffffff;
}
.sidebar-content h1, .sidebar-content h2, .sidebar-content h3, .sidebar-content p, .sidebar-content li {
    color: #f5f5f5 !important;
}
.stMarkdown, h1, h2, h3, p {
    color: #1a237e;
}
.stButton > button {
    background-color: #3949ab;
    color: white;
    border-radius: 8px;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #5c6bc0;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ===========================
# 3. Title & Header
# ===========================
st.title("🎯 AI Vision App")
st.markdown("### Example: Object Extraction")

sample_img = "sample_extraction.jpg"
if os.path.exists(sample_img):
    st.image(sample_img, use_container_width=True)
else:
    st.warning("⚠️ Sample image not found! Please add `sample_extraction.jpg` to your repo.")

# ===========================
# 4. Sidebar Info
# ===========================
st.sidebar.markdown(
    """
    <div class="sidebar-content">
    <h2>📘 About This App</h2>
    <p><b>AI Vision Extraction</b> allows you to automatically isolate the main object 
    from any image using an advanced <b>DeepLabV3+</b> segmentation model.</p>

    <h3>⏱️ How to Use</h3>
    <ol>
        <li>Upload an image (JPG, JPEG, or PNG).</li>
        <li>The AI extracts the main object and removes the background.</li>
        <li>Preview and download your clean extracted image.</li>
    </ol>

    <p>💡 Works best with clear images where the main subject is distinct.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ===========================
# 5. Load Model
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
# 6. Helper Functions
# ===========================
def preprocess_image(image: Image.Image):
    img = image.convert("RGB")
    img = np.array(img)
    img_resized = cv2.resize(img, (256, 256))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return img_tensor, img

def predict_mask(model, device, image: Image.Image):
    img_tensor, orig = preprocess_image(image)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        output = model(img_tensor)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
    mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))
    mask = (mask > 0.5).astype(np.uint8)
    result = orig * mask[:, :, np.newaxis]
    return result.astype(np.uint8)

def convert_to_bytes(image_np):
    img = Image.fromarray(image_np)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ===========================
# 7. Upload & Predict
# ===========================
st.header("🧠 Try It Yourself")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    input_img = Image.open(uploaded_file)
    result = predict_mask(model, device, input_img)

    col1, col2 = st.columns(2)
    with col1:
        st.image(input_img, caption="Original Image", use_container_width=True)
    with col2:
        st.image(result, caption="Extracted Object", use_container_width=True)

    # Download button
    img_bytes = convert_to_bytes(result)
    st.download_button(
        label="📥 Download Extracted Image",
        data=img_bytes,
        file_name="extracted_object.png",
        mime="image/png"
    )
