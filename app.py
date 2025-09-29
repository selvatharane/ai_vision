import streamlit as st
import torch
import segmentation_models_pytorch as smp
import torchvision.transforms as T
import numpy as np
from PIL import Image
import cv2

# ================================
# 1. Load Model Once
# ================================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(device)
    model.load_state_dict(torch.load("best_deeplab.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# ================================
# 2. Preprocessing
# ================================
def preprocess_image(image):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# ================================
# 3. TTA Prediction
# ================================
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

# ================================
# 4. Post-processing
# ================================
def postprocess_mask(mask_tensor, orig_size):
    mask = mask_tensor.squeeze().cpu().detach().numpy()
    mask = (mask * 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.resize(mask, orig_size)
    return mask

# ================================
# 5. Streamlit UI
# ================================
st.set_page_config(
    page_title="AI Vision Extraction",
    page_icon="🖥️",
    layout="wide"
)

st.title("🖥️ AI Vision Object Extraction")
st.markdown(
    "Upload an image, and the app will automatically extract objects using **DeepLabV3+** with **TTA & post-processing**."
)

# Upload
uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    orig_size = img.size
    st.image(img, caption="Original Image", use_column_width=True)

    with st.spinner("Processing..."):
        input_tensor = preprocess_image(img).to(device)
        mask_tensor = predict_with_tta(model, input_tensor)
        mask_np = postprocess_mask(mask_tensor, orig_size)

        # Overlay
        overlay = np.array(img)
        overlay[mask_np == 0] = 0

    st.success("✅ Extraction complete!")

    # Attractive visualization with two columns
    col1, col2, col3 = st.columns(3)
    col1.image(img, caption="Original Image", use_column_width=True)
    col2.image(mask_np, caption="Segmentation Mask", use_column_width=True)
    col3.image(overlay, caption="Extracted Objects", use_column_width=True)

# Optional footer
st.markdown("---")
st.markdown("Built with 💡 **DeepLabV3+**, **PyTorch**, and **Streamlit**")
