import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from brain_train import SegNet  # rename to match your actual model file

st.set_page_config(page_title="Brain Tumor Diagnosis", layout="centered")
st.title("🧠 Brain Tumor Segmentation")
st.markdown("Upload a brain MRI scan to detect tumor regions using a trained ML model.")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegNet().to(device)
model.load_state_dict(torch.load("segnet_braincell.pth", map_location=device))  # path to your trained model
model.eval()

# Upload input
uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original MRI", use_column_width=True)

    # Convert to grayscale and resize
    img_resized = image.resize((256, 256))
    img_gray = np.array(img_resized.convert("L")) / 255.0
    img_tensor = torch.tensor(img_gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = output.squeeze().cpu().numpy()
        binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    # Apply mask to color image
    segmented = np.array(img_resized)
    segmented[binary_mask == 0] = 0

    st.markdown("### 🧪 Segmentation Results")
    col1, col2, col3 = st.columns(3)
    col1.image(img_resized, caption="Resized Image")
    col2.image(binary_mask, caption="Predicted Mask", clamp=True)
    col3.image(segmented, caption="Tumor Highlighted")

    result_image = Image.fromarray(segmented)
    st.download_button("📥 Download Segmented Output", result_image.tobytes(), file_name="segmented_output.png")
