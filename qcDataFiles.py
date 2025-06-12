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
import argparse

import os
import csv
from pathlib import Path

import pickle 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dir = Path("/work3/kvabo/PKGCTORG/newCT")

class CTOrganSegDataset(Dataset):
    def __init__(self, case_numbers, dir, filteredData=Path("Thesis/liverSlices.pkl")):
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
        target_class = 1
        binary_mask = (label_slice == target_class).astype(np.uint8)

        # Resize mask
        mask_pil = Image.fromarray(binary_mask * 255)
        mask_resized = TF.resize(mask_pil, self.resize_size)
        mask_array = np.array(mask_resized) / 255.0

        # Stack input and mask
        X = np.stack([volume_array, mask_array], axis=0).astype(np.float32)

        return torch.tensor(X), case_number, slice_idx

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


model = UNet()

allCases = list(range(140))

# Create datasets
allDataset = CTOrganSegDataset(allCases, dir)

batch_size = 10

# Create DataLoaders
allLoader = DataLoader(allDataset, batch_size=batch_size, shuffle=False)

diceLoss = smp.losses.DiceLoss(mode='binary', from_logits=False, eps=1e-7, smooth = 1)


parser = argparse.ArgumentParser(description='Trainde QC dataset model')
parser.add_argument('--modelPath', type=str, required=True, help='Path to pretrained model (e.g. qcKidneyDsc0.1Continued.pth')
args = parser.parse_args()

model_path = Path(args.modelPath)
model_name = model_path.stem

# Load model
model = UNet()
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.to(device)
model.eval()

results = []
model_name = os.path.splitext(os.path.basename(model_path))[0]
save_dir = Path("/work3/kvabo/predictions") / model_name
save_dir.mkdir(parents=True, exist_ok=True)
results_csv = save_dir / "results.csv"

with torch.no_grad():
    for batch_idx, (X, case_ids, slice_indices) in enumerate(allLoader):
        inputs = X[:, 0:1, :, :].to(device)
        labels = X[:, 1:, :, :].to(device)

        outputs = torch.sigmoid(model(inputs))
        preds_bin = (outputs > 0.5).float().cpu()

        for i in range(inputs.shape[0]):
            pred_mask = preds_bin[i, 0].numpy()
            dsc_pred = 1 - diceLoss(preds_bin[i:i+1].to(device), labels[i:i+1]).item()
            case_id = case_ids[i].item()
            slice_idx = int(slice_indices[i])
            pred_filename = f"{case_id}_slice_{slice_idx}_pred.npy"
            pred_path = save_dir / pred_filename
            np.save(pred_path, pred_mask)

            results.append({
                "case": case_id,
                "slice": slice_idx,
                "model": model_name,
                "dsc": round(dsc_pred, 5),
                "prediction_file": pred_filename
            })

# Save results CSV
with open(results_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["case", "slice", "model", "dsc", "prediction_file"])
    writer.writeheader()
    writer.writerows(results)

print(f"Saved predictions to {save_dir}")
print(f"Saved CSV summary to {results_csv}")
