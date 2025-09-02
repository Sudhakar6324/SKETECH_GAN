# app.py
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import UNet

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load generator
gen = UNet(3, 3).to(device)
gen.load_state_dict(torch.load("gen_best_ssim.pth", map_location=device))
gen.eval()

# Transform (must match training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def denorm(tensor):
    tensor = (tensor * 0.5 + 0.5).clamp(0,1)
    return transforms.ToPILImage()(tensor)

# Streamlit UI
st.title("🎨 Sketch → Photo Converter (Pix2Pix)")
uploaded_file = st.file_uploader("Upload a sketch (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    sketch = Image.open(uploaded_file).convert("RGB")
    st.image(sketch, caption="🖼️ Uploaded Sketch", width=256)

    # Run model
    sketch_t = transform(sketch).unsqueeze(0).to(device)
    with torch.no_grad():
        fake = gen(sketch_t)

    fake_img = denorm(fake.squeeze().cpu())
    st.image(fake_img, caption="✨ Generated Photo", width=256)

    # Download button
    fake_img.save("generated.png")
    with open("generated.png", "rb") as f:
        st.download_button("⬇️ Download Generated Image", f, "generated.png")
