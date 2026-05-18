"""
STEP 3 — Train the MobileNetV2 road segmentation model

Expected folder structure:
  data/
    images/
    masks/
"""

import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import os
import cv2

# Config
IMAGE_DIR   = 'data/images'
MASK_DIR    = 'data/masks'
MODEL_PATH  = 'road_segmenter.pth'
IMG_SIZE    = (256, 256)
BATCH_SIZE  = 16
EPOCHS      = 40
LR          = 0.005
VAL_SPLIT   = 0.15          # 15% of data used for validation

# Dataset
class WRCDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size=IMG_SIZE, augment=False):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.size      = size
        self.augment   = augment

        all_images = sorted(os.listdir(image_dir))
        self.images = []
        for img_name in all_images:
            base = img_name.replace('.jpg', '').replace('.png', '')
            mask_name = f'{base}_mask.png'
            if os.path.exists(os.path.join(mask_dir, mask_name)):
                self.images.append(img_name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.size)
        img = np.array(img, dtype=np.float32) / 255.0

        base = self.images[idx].replace('.jpg', '').replace('.png', '')
        mask_path = os.path.join(self.mask_dir, f'{base}_mask.png')
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize(self.size)
        mask = np.array(mask, dtype=np.float32) / 255.0

        # augmentation — only during training
        if self.augment:
            # horizontal flip
            if np.random.rand() > 0.5:
                img  = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()

            # random brightness
            img = np.clip(img * np.random.uniform(0.6, 1.4), 0, 1)

            # random contrast
            mean = img.mean()
            img  = np.clip((img - mean) * np.random.uniform(0.8, 1.2) + mean, 0, 1)

            # random blur (simulates motion blur in fast corners)
            if np.random.rand() > 0.7:
                k   = np.random.choice([3, 5])
                img = cv2.GaussianBlur(img, (k, k), 0)

        img  = torch.FloatTensor(img).permute(2, 0, 1)
        mask = torch.FloatTensor(mask)
        return img, mask


# MobileNetV2 encoder + decoder
class RoadSegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights='DEFAULT')
        self.encoder = mobilenet.feature

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        mask     = self.decoder(features)
        return mask


# Training
def train():

    # check for GPU — training is much faster with one
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    full_dataset = WRCDataset(IMAGE_DIR, MASK_DIR)

    if len(full_dataset) < 20:
        print(f'ERROR: Only {len(full_dataset)} labeled frames found.')
        print('You need at least 20 labeled frames. Aim for 150-200.')
        return

    # split into train and validation
    val_size   = max(1, int(len(full_dataset) * VAL_SPLIT))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset.augment = True


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    print(f'Training on {train_size} frames, validating on {val_size} frames')

    # model, optimizer, loss
    model     = RoadSegmenter().to(device)
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=LR)
    loss_fn   = nn.BCELoss()

    best_val_loss = float('inf')

    print(f'\nTraining for {EPOCHS} epochs...\n')

    train_losses = []
    val_losses   = []

    for epoch in range(EPOCHS):
        # train
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs).squeeze(1)
            # resize prediction to match mask if needed
            if preds.shape != masks.shape:
                preds = torch.nn.functional.interpolate(
                    preds.unsqueeze(1), size=masks.shape[1:], mode='bilinear'
                ).squeeze(1)

            loss = loss_fn(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs  = imgs.to(device)
                masks = masks.to(device)
                preds = model(imgs).squeeze(1)
                if preds.shape != masks.shape:
                    preds = torch.nn.functional.interpolate(
                        preds.unsqueeze(1), size=masks.shape[1:], mode='bilinear'
                    ).squeeze(1)
                val_loss += loss_fn(preds, masks).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)

        # save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), MODEL_PATH)
            saved_marker = ' ← saved'
        else:
            saved_marker = ''

        print(f'Epoch {epoch+1:02d}/{EPOCHS}  '
              f'train loss: {avg_train:.4f}  '
              f'val loss: {avg_val:.4f}{saved_marker}')
        
        train_losses.append(avg_train)
        val_losses.append(avg_val)

    print(f'\n Best model saved to {MODEL_PATH}')

    epochs_range = list(range(1, len(train_losses) + 1))
    best_epoch   = val_losses.index(min(val_losses)) + 1
    best_val     = min(val_losses)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_losses, label='Train Loss', color='blue',   linewidth=2)
    plt.plot(epochs_range, val_losses,   label='Val Loss',   color='orange', linewidth=2)
    plt.axvline(x=best_epoch, color='green', linestyle='--',
               linewidth=1.5, label=f'Best model (epoch {best_epoch})')
    plt.scatter([best_epoch], [best_val], color='green', zorder=5, s=80)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150)
    print(f'Loss graph saved to training_loss.png')

if __name__ == '__main__':
    train()
