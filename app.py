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
# 2. Custom Styling
# ===========================
custom_style = """
<style>
/* Main page background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f3f7ff 0%, #dbe9ff 50%, #f3f7ff 100%);
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    color: #ffffff;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Sidebar close/hamburger button */
[data-testid="stSidebarNav"] button, [data-testid="stSidebarCollapseControl"] button {
    color: white !important;
    filter: brightness(1.3);
}

/* Headers and titles */
h1, h2, h3, h4 {
    color: #1e293b !important;
}

/* Image shadow */
img {
    border-radius: 12px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.15);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    box-shadow: 0px 3px 10px rgba(59,130,246,0.4);
    border: none;
    transition: all 0.3s ease-in-out;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #2563eb);
    transform: scale(1.05);
}

/* Markdown content */
.block-container {
    padding-top: 1.5rem;
}

/* Glass card container for header section */
.glass-card {
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(8px);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
}
</style>
"""
st.markdown(custom_style, unsafe_allow_html=True)

# ===========================
# 3. App Title Section
# ===========================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.title("✨ AI Vision Extraction")
st.markdown("#### Example: Object Extraction")
st.markdown('</div>', unsafe_allow_html=True)

# Show sample image (smaller)
sample_img = "sample_extraction.jpg"
if os.path.exists(sample_img):
    st.image(sample_img, caption=None, use_container_width=False, width=500)
else:
    st.warning("⚠️ Sample image not found! Please add `sample_extraction.jpg` to your repo.")

# ===========================
# 4. Sidebar Content
# ===========================
st.sidebar.markdown(
    """
    <div style='padding: 0.5rem;'>
        <h2>📘 About This App</h2>
        <p><b>AI Vision Extraction</b> helps you automatically isolate the main object 
        from any image using an advanced <b>DeepLabV3+</b> segmentation model.</p>

        <h3>🪄 How to Use</h3>
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
st.markdown("### 🧠 Try It Yourself")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    input_img = Image.open(uploaded_file)
    result = predict_mask(model, device, input_img)

    col1, col2 = st.columns(2)
    with col1:
        st.image(input_img, caption="Original Image", use_container_width=True)
    with col2:
        st.image(result, caption="Extracted Object", use_container_width=True)

    img_bytes = convert_to_bytes(result)
    st.download_button(
        label="📥 Download Extracted Image",
        data=img_bytes,
        file_name="extracted_object.png",
        mime="image/png"
    )
