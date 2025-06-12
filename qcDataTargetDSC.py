############## Imports ##############
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import nibabel as nib
import matplotlib.pyplot as plt

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
import torch.nn.init as init

# Image preprocessing
from skimage.transform import resize

# Model and training utilities
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split

# Setup
import logging
import pickle 
import argparse

parser = argparse.ArgumentParser(description='Quality Control Dataset')
parser.add_argument('--targetDSC', type=float, required=True, help='Target Dice Similarity Coefficient (e.g., 0.3)')

args = parser.parse_args()

targetDSC = args.targetDSC

############## Data Paths and Logging ##############
logging.basicConfig(
    filename=f'qcBoneDSC{targetDSC}.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

path = Path().cwd()

dir = Path("/work3/kvabo/PKGCTORG/newCT")


############## Data Loading ##############

class CTOrganSegDataset(Dataset):
    def __init__(self, case_numbers, dir, filteredData=Path("Thesis/boneSlices.pkl")):
        self.case_numbers = case_numbers
        self.data_dir = Path(dir)
        self.resize_size = (256, 256)

        # Load precomputed slice data
        with open(filteredData, 'rb') as f:
            all_slice_data = pickle.load(f)

        # Keep only slices for relevant cases
        self.slice_data = [
            (case_number, slice_idx)
            for (case_number, slice_idx) in all_slice_data
            if case_number in self.case_numbers
        ]

        # Preload volume and label paths
        self.volumePaths = {
            case_number: self.data_dir / f"volume-{case_number}.npy"
            for case_number in self.case_numbers
        }
        self.labelPaths = {
            case_number: self.data_dir / f"labels-{case_number}.npy"
            for case_number in self.case_numbers
        }

    def __len__(self):
        return len(self.slice_data)

    def __getitem__(self, idx):
        case_number, slice_idx = self.slice_data[idx]
        volume = np.load(self.volumePaths[case_number])
        label = np.load(self.labelPaths[case_number])

        volume_slice = volume[:, :, slice_idx]
        label_slice = label[:, :, slice_idx]

        # Resize CT slice
        volume_pil = Image.fromarray(volume_slice).convert("F")
        volume_resized = TF.resize(volume_pil, self.resize_size)
        volume_array = np.array(volume_resized, dtype=np.float32)

        # Determine which class to segment
        target_class = 5
        binary_mask = (label_slice == target_class).astype(np.uint8)

        # Resize mask
        mask_pil = Image.fromarray(binary_mask * 255)
        mask_resized = TF.resize(mask_pil, self.resize_size)
        mask_array = np.array(mask_resized) / 255.0

        # Stack input and mask
        X = np.stack([volume_array, mask_array], axis=0).astype(np.float32)

        return torch.tensor(X), case_number, slice_idx



# Define case splits
train_cases = list(range(140))  
# Create datasets
train_dataset = CTOrganSegDataset(train_cases, dir)

batch_size = 40

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


############## Model Setup ##############

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc_conv0 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2)  # 256 -> 128
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)  # 128 -> 64
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)  # 64 -> 32
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)  # 32 -> 16

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # Decoder
        self.upsample0 = nn.Upsample(scale_factor=2)  # 16 -> 32
        self.dec_conv0 = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2)  # 32 -> 64
        self.dec_conv1 = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2)  # 64 -> 128
        self.dec_conv2 = nn.Conv2d(64 + 64, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2)  # 128 -> 256
        self.dec_conv3 = nn.Conv2d(64 + 64, 64, 3, padding=1)

        # Final output
        self.final_conv = nn.Conv2d(64, out_channels, 1)  # [B, 64, 256, 256] -> [B, 4, 256, 256]


    def forward(self, x):
        # Encoder
        s0 = F.relu(self.enc_conv0(x))   # [B, 64, 256, 256]
        e0 = self.pool0(s0)              # [B, 64, 128, 128]
        s1 = F.relu(self.enc_conv1(e0))  # [B, 64, 128, 128]
        e1 = self.pool1(s1)              # [B, 64, 64, 64]
        s2 = F.relu(self.enc_conv2(e1))  # [B, 64, 64, 64]
        e2 = self.pool2(s2)              # [B, 64, 32, 32]
        s3 = F.relu(self.enc_conv3(e2))  # [B, 64, 32, 32]
        e3 = self.pool3(s3)              # [B, 64, 16, 16]

        # Bottleneck
        b = F.relu(self.bottleneck_conv(e3))  # [B, 64, 16, 16]

        # Decoder
        d0 = F.relu(self.dec_conv0(torch.cat([self.upsample0(b), s3], dim=1)))  # [B, 64, 32, 32]
        d1 = F.relu(self.dec_conv1(torch.cat([self.upsample1(d0), s2], dim=1))) # [B, 64, 64, 64]
        d2 = F.relu(self.dec_conv2(torch.cat([self.upsample2(d1), s1], dim=1))) # [B, 64, 128, 128]
        d3 = F.relu(self.dec_conv3(torch.cat([self.upsample3(d2), s0], dim=1))) # [B, 64, 256, 256]

        out = self.final_conv(d3)  # [B, 4, 256, 256]
        return out

# He initialization function
def init_he(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)



model = UNet()
model.apply(init_he)


############## Loss Functions ##############
diceLoss = smp.losses.DiceLoss(mode='binary', from_logits=False, eps=1e-7, smooth = 1)

############## Model Training Specifications ##############
num_epochs = 15
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.7)

############## Training Loop ##############

model = model.to(device)

for epoch in range(num_epochs):
    logger.info(f"Epoch {epoch + 1}/{num_epochs} - Training Started")
    model.train()
    train_loss = 0.0

    for batch, _, _ in train_loader:
        inputs = batch[:, 0:1].to(device)
        labels = batch[:, 1:].to(device)

        opt.zero_grad()

        outputs = torch.sigmoid(model(inputs))
        loss = torch.abs(diceLoss(labels, outputs) - (1-targetDSC))
        loss.backward()
        opt.step()
        train_loss += loss.item()

        
    scheduler.step()

    logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")


    torch.save(model.state_dict(), path / f"qcBoneDsc{targetDSC}.pth")
    

logger.info("Training completed.")
