import os
import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
# ------------------------------
# Dataset
# ------------------------------
class Sketch2PhotoDataset(Dataset):
    def __init__(self, sketch_dir, photo_dir, transform=None):
        self.sketch_paths = sorted(glob.glob(os.path.join(sketch_dir, "*.jpg")))
        self.photo_paths = sorted(glob.glob(os.path.join(photo_dir, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return min(len(self.sketch_paths), len(self.photo_paths))

    def __getitem__(self, idx):
        sketch = Image.open(self.sketch_paths[idx]).convert("RGB")
        photo = Image.open(self.photo_paths[idx]).convert("RGB")
        if self.transform:
            sketch = self.transform(sketch)
            photo = self.transform(photo)
        return sketch, photo

# ------------------------------
# Generator (U-Net)
# ------------------------------
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        self.encoder = nn.ModuleList([
            self.block(in_channels, features, normalize=False),
            self.block(features, features*2),
            self.block(features*2, features*4),
            self.block(features*4, features*8),
            self.block(features*8, features*8),
            self.block(features*8, features*8),
        ])
        self.decoder = nn.ModuleList([
            self.block(features*8, features*8, transposed=True, dropout=True),
            self.block(features*16, features*8, transposed=True, dropout=True),
            self.block(features*16, features*4, transposed=True),
            self.block(features*8, features*2, transposed=True),
            self.block(features*4, features, transposed=True),
        ])
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def block(self, in_channels, out_channels, normalize=True, transposed=False, dropout=False):
        layers = []
        if transposed:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2) if not transposed else nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        encs = []
        for enc in self.encoder:
            x = enc(x)
            encs.append(x)
        for i, dec in enumerate(self.decoder):
            x = dec(x)
            x = torch.cat([x, encs[-(i+2)]], dim=1)
        return self.final(x)

# ------------------------------
# Discriminator (PatchGAN)
# ------------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6, features=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features*2, 4, 2, 1),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features*2, features*4, 4, 2, 1),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features*4, features*8, 4, 1, 1),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features*8, 1, 4, 1, 1)
        )

    def forward(self, sketch, photo):
        return self.model(torch.cat([sketch, photo], dim=1))

# ------------------------------
# Gradient Penalty
# ------------------------------
def compute_gradient_penalty(disc, real, fake, condition, device):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    pred = disc(condition, interpolated)
    grads = torch.autograd.grad(
        outputs=pred,
        inputs=interpolated,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grads = grads.view(grads.size(0), -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# ------------------------------
# Validation
# ------------------------------
def validate(gen, val_loader, device):
    gen.eval()
    total_psnr, total_ssim, count = 0, 0, 0
    with torch.no_grad():
        for sketches, photos in val_loader:
            sketches, photos = sketches.to(device), photos.to(device)
            fake_photos = gen(sketches)
            fake_np = fake_photos.cpu().numpy().transpose(0, 2, 3, 1)
            real_np = photos.cpu().numpy().transpose(0, 2, 3, 1)
            for f, r in zip(fake_np, real_np):
                f = np.clip((f + 1) / 2, 0, 1)
                r = np.clip((r + 1) / 2, 0, 1)
                total_psnr += peak_signal_noise_ratio(r, f, data_range=1)
                total_ssim += structural_similarity(r, f, channel_axis=-1, data_range=1)
                count += 1
    gen.train()
    return total_psnr / count, total_ssim / count

# ------------------------------
# Training Loop
from torch.amp import autocast, GradScaler

import torchvision.utils as vutils

def train_bce_gp(
    gen, disc,
    train_loader, val_loader,
    opt_G, opt_D,
    criterion_GAN, criterion_L1,
    lambda_L1=100, lambda_gp=10,
    device="cuda", n_epochs=100
):
    best_ssim = -1
    scaler_G = GradScaler(device)
    scaler_D = GradScaler(device)

    for epoch in tqdm(range(n_epochs)):
        for i, (sketches, photos) in enumerate(train_loader):
            sketches, photos = sketches.to(device), photos.to(device)

            # --- Train Discriminator ---
            opt_D.zero_grad()
            with autocast(device_type="cuda"):
                fake_photos = gen(sketches)

                pred_real = disc(sketches, photos)
                pred_fake = disc(sketches, fake_photos.detach())

                loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
                loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
                loss_D = 0.5 * (loss_D_real + loss_D_fake)

                #gp = compute_gradient_penalty(disc, photos, fake_photos.detach(), sketches, device)
                loss_D = loss_D 
            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

            # --- Train Generator ---
            opt_G.zero_grad()
            with autocast(device_type="cuda"):
                pred_fake = disc(sketches, fake_photos)

                loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
                loss_L1 = criterion_L1(fake_photos, photos) * lambda_L1
                loss_G = loss_GAN + loss_L1

            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G)
            scaler_G.update()

        print(f"[Epoch {epoch+1}/{n_epochs} | Batch {i}] "
              f"Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}", flush=True)

        # --- Validate at epoch end ---
        val_psnr, val_ssim = validate(gen, val_loader, device)
        print(f"Epoch {epoch+1}: Val PSNR {val_psnr:.2f}, Val SSIM {val_ssim:.4f}", flush=True)

        # --- Save visual samples every 25 epochs ---
        if (epoch + 1) % 50 == 0:
            gen.eval()
            with torch.no_grad():
                v_sketch, v_photo = next(iter(val_loader))
                v_sketch, v_photo = v_sketch.to(device), v_photo.to(device)
                v_fake = gen(v_sketch)
                vutils.save_image(
                    torch.cat([v_sketch, v_photo, v_fake], 0),
                    f"val_sample_epoch{epoch+1}.png",
                    normalize=True, nrow=3
                )
            gen.train()

        # --- Save best checkpoint ---
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save(gen.state_dict(), "best_generator.pth")
            print(f"? Saved best model (SSIM={best_ssim:.4f})", flush=True)
# Main
# ------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 2e-4
    n_epochs = 200
    batch_size = 1

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = Sketch2PhotoDataset("train/sketches", "train/photos", transform=transform)
    val_dataset = Sketch2PhotoDataset("val/sketches", "val/photos", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,num_workers=8, pin_memory=True)

    gen = UNetGenerator().to(device)
    disc = PatchDiscriminator().to(device)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    opt_G = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    train_bce_gp(
        gen, disc,
        train_loader, val_loader,
        opt_G, opt_D,
        criterion_GAN, criterion_L1,
        lambda_L1=100, lambda_gp=10,
        device=device, n_epochs=n_epochs
    )
