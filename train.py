import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.unet import get_unet
from ddpm import ForwardDiffusion
import os

# Config
device = torch.device("metal" if torch.cuda.is_available() else "cpu")
T = 1000
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4
SAVE_PATH = "ddpm_fashionmnist.pth"

# Model config
cfg = {
    "in_channels": 1,
    "out_channels": 1,
    "time_emb_dim": 128,
    "use_conv": True
}

# Load model
model = get_unet(cfg).to(device)

# Forward diffusion scheduler
forward_diffusion = ForwardDiffusion(timesteps=T)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)  # scale to [-1, 1]
])

dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for i, (x, _) in enumerate(dataloader):
        x = x.to(device)
        t = torch.randint(0, T, (x.size(0),), device=device).long()
        noise = torch.randn_like(x)

        x_t = forward_diffusion.noising(x, noise, t)
        noise_pred = model(x_t, t)

        loss = F.mse_loss(noise_pred, noise)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch+1}] complete. Avg Loss: {epoch_loss / len(dataloader):.4f}")

    # Save model checkpoint
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Checkpoint saved to {SAVE_PATH}")
