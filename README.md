# SKETECH_GAN

This project implements a GAN-based approach for sketch generation using a **U-Net generator** and a **patched discriminator**. Additionally, a simple **Streamlit interface** is built for easy interaction and testing.

## Features

- **Generator:** U-Net architecture  
- **Discriminator:** PatchGAN (patched discriminator)  
- **Interface:** Interactive Streamlit app for testing and visualization  
- **Performance:** Achieved a **validation SSIM of 0.085** on the validation dataset  
## Interface

You can interact with the model using the Streamlit interface:

![Streamlit Interface](assets/Screenshot 2025-09-03 033640.png)

## Installation

Clone the repository:

```bash
git clone https://github.com/Sudhakar6324/SKETECH_GAN.git
cd SKETECH_GAN
