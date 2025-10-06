import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2
import os
from io import BytesIO

# ===========================
# 1. Page Configuration
# ===========================
st.set_page_config(page_title="AI Vision Extraction", layout="wide")

# --- Custom Background Style ---
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #d7e1ec, #ffffff, #e3f2fd);
    background-attachment: fixed;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3c72, #2a5298);
    color: white;
}
h1, h2, h3, h4, h5, h6 {
    color: #102542 !important;
}
div.stButton > button {
    background-color: #1976d2;
    color: white;
    border-radius: 12px;
    height: 2.8em;
    width: 100%;
    font-size: 1em;
}
div.stButton > button:hover {
    background-color: #1565c0;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ===========================
# 2. App Header
# ===========================
st.title("🧠 AI Vision Extraction App")
st.caption("DeepLabV3+ powered object extraction using AI segmentation")

# ===========================
# 3. Example Image
# ===========================
st.markdown("### 🖼️ Example: Object Extraction")

sample_img = "sample.png"
if os.path.exists(sample_img):
    st.image(sample_img, use_container_width=True)
else:
    st.warning("⚠️ Add a 'sample_extraction.jpg' in your repo to show a preview here.")

# ===========================
# Sidebar Instructions (Simplified)
# ===========================
st.sidebar.title("📘 About This App")
st.sidebar.info("""
**AI Vision Extraction** allows you to automatically isolate the main object from any image 
using an advanced DeepLabV3+ segmentation model.

### 🧭 How to Use
1. Upload an image (JPG, JPEG, or PNG).  
2. The AI extracts the main object and removes the background.  
3. Adjust the mask strength if needed.  
4. Download your clean, extracted object image.

💡 Works best with clear images where the main subject is distinct.
""")


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
def preprocess_image(image):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# --- TTA Inference ---
def predict_with_tta(model, input_tensor):
    aug_list = [
        lambda x: x,
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(x, [2]),
        lambda x: torch.rot90(x, k=1, dims=[2, 3]),
        lambda x: torch.rot90(x, k=3, dims=[2, 3])
    ]
    outputs = []
    for f in aug_list:
        t = f(input_tensor)
        with torch.no_grad():
            out = model(t)
            out = torch.sigmoid(out)
            if f != aug_list[0]:
                if f == aug_list[1]: out = torch.flip(out, [3])
                elif f == aug_list[2]: out = torch.flip(out, [2])
                elif f == aug_list[3]: out = torch.rot90(out, k=3, dims=[2, 3])
                elif f == aug_list[4]: out = torch.rot90(out, k=1, dims=[2, 3])
            outputs.append(out)
    return torch.mean(torch.stack(outputs), dim=0)

# --- Postprocessing ---
def postprocess_mask(mask_tensor, threshold_factor=1.0):
    mask = mask_tensor.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (mask * threshold_factor).clip(0, 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

# --- Extraction ---
def extract_object(image, mask):
    image_np = np.array(image.convert("RGB"))
    mask_resized = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
    output = image_np.copy()
    output[mask_resized == 0] = [0, 0, 0]
    return output

# ===========================
# 7. Upload & Predict
# ===========================
st.markdown("---")
st.header("🧩 Upload Your Image")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
mask_strength = st.slider("🎛️ Adjust Mask Strength", 0.5, 1.5, 1.0, 0.1)
compare_view = st.checkbox("Show Comparison View (Side-by-side)", value=True)

if uploaded_file is not None and model is not None:
    input_img = Image.open(uploaded_file).convert("RGB")

    with st.spinner("🔍 Extracting object... Please wait..."):
        input_tensor = preprocess_image(input_img).to(device)
        mask_pred = predict_with_tta(model, input_tensor)
        mask_np = postprocess_mask(mask_pred, threshold_factor=mask_strength)
        extracted = extract_object(input_img, mask_np)

    st.success("✅ Extraction complete!")

    # Display output
    if compare_view:
        col1, col2 = st.columns(2)
        with col1:
            st.image(input_img, caption="Original Image", use_container_width=True)
        with col2:
            st.image(extracted, caption="Extracted Object", use_container_width=True)
    else:
        st.image(extracted, caption="Extracted Object", use_container_width=True)

    # Extra info
    st.caption(f"📏 Original Image Size: {input_img.size[0]}x{input_img.size[1]}")

    # Download button
    result_img = Image.fromarray(extracted)
    buf = BytesIO()
    result_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label="📥 Download Extracted Image",
        data=byte_im,
        file_name="extracted_object.png",
        mime="image/png"
    )

st.markdown("---")
st.markdown("🔬 *Powered by DeepLabV3+ and PyTorch — built with ❤️ using Streamlit*")
