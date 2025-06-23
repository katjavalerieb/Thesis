import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import nibabel as nib
import matplotlib.pyplot as plt
import csv

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import functional as TF
import torch.nn.init as init
from torchsummary import summary

# Image preprocessing
from skimage.transform import resize

# Model and training utilities
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split

# Setup
import logging
import pickle 
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dir = Path("/work3/kvabo/PKGCTORG/newCT")

parser = argparse.ArgumentParser(description='Automatic Quality Control Model Training')
parser.add_argument('--organ', type=str, required=True, help='Liver, Bone or Kidney')

args = parser.parse_args()

organ = args.organ

logging.basicConfig(
    filename=f'aqc{organ}Evalshort.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_conv = self.conv(x)
        x_pool = self.pool(x_conv)
        return x_pool, x_conv

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            nn.BatchNorm2d(out_channels + skip_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((skip, x), dim=1)
        return self.block(x)

class QCUNet(nn.Module):
    def __init__(self, in_channels=2):
        super(QCUNet, self).__init__()
        self.enc1 = EncoderBlock(in_channels, 40)
        self.enc2 = EncoderBlock(40, 40)
        self.enc3 = EncoderBlock(40, 80)
        self.enc4 = EncoderBlock(80, 160)
        self.enc5 = EncoderBlock(160, 320)

        self.bottleneck = ConvBlock(320, 640)

        self.dec5 = DecoderBlock(640, 320, 320)
        self.dec4 = DecoderBlock(320, 160, 160)
        self.dec3 = DecoderBlock(160, 80, 80)
        self.dec2 = DecoderBlock(80, 40, 40)
        self.dec1 = DecoderBlock(40, 40, 40)

        self.output_conv = nn.Conv2d(40, 2, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Outputs shape (B, 1, 1, 1)

    def forward(self, x):
        x1_pool, x1 = self.enc1(x)
        x2_pool, x2 = self.enc2(x1_pool)
        x3_pool, x3 = self.enc3(x2_pool)
        x4_pool, x4 = self.enc4(x3_pool)
        x5_pool, x5 = self.enc5(x4_pool)

        x = self.bottleneck(x5_pool)

        x = self.dec5(x, x5)
        x = self.dec4(x, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)

        x = self.output_conv(x)                    # (B, 1, H, W)
        x = self.global_avg_pool(x)                # (B, 1, 1, 1)
        return x.view(x.size(0), -1)               # Flatten to shape (B, 1)


model = QCUNet()
model_path = f"Thesis/aqcModel{organ}V4.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.to(device)
model.eval()

class QCDataset(Dataset):
    def __init__(self, qc_dirs, ct_dir):
        self.ct_dir = Path(ct_dir)
        self.data = []

        # Load and combine all resultsUpdate.csv files
        for qc_dir in qc_dirs:
            qc_path = Path(qc_dir)
            df = pd.read_csv(qc_path / "resultsUpdate.csv")
            df["qc_dir"] = qc_path  # Track where each sample comes from
            self.data.append(df)

        self.data = pd.concat(self.data, ignore_index=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        case_number = int(row['case'])
        slice_idx = int(row['slice'])
        qc_dir = Path(row['qc_dir'])

        # Target vector: [DSC, MVSF]
        Y = torch.tensor([row['dsc'], row['mvsf']], dtype=torch.float32)

        # Load segmentation
        seg_path = qc_dir / f"{case_number}_slice_{slice_idx}_pred.npy"
        seg = np.load(seg_path).astype(np.float32)
        seg_tensor = torch.from_numpy(seg).unsqueeze(0)  # shape: [1, 256, 256]

        # Load and resize CT slice
        ct_path = self.ct_dir / f"volume-{case_number}.npy"
        ct = np.load(ct_path)[:, :, slice_idx].astype(np.float32)
        ct_tensor = torch.from_numpy(ct).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]
        ct_resized = F.interpolate(ct_tensor, size=(256, 256), mode='bilinear', align_corners=False)
        ct_resized = ct_resized.squeeze(0)  # shape: [1, 256, 256]

        # Stack CT and segmentation
        X = torch.cat([ct_resized, seg_tensor], dim=0)  # shape: [2, 256, 256]

        return X, Y, case_number, slice_idx, str(qc_dir)

qc_dirs = [
    f"/work3/kvabo/predictions/qc{organ}Dsc1.0",
    f"/work3/kvabo/predictions/qc{organ}Dsc0.9",
    f"/work3/kvabo/predictions/qc{organ}Dsc0.8",
    f"/work3/kvabo/predictions/qc{organ}Dsc0.7",
    f"/work3/kvabo/predictions/qc{organ}Dsc0.6",
    f"/work3/kvabo/predictions/qc{organ}Dsc0.5",
    f"/work3/kvabo/predictions/qc{organ}Dsc0.4",
    f"/work3/kvabo/predictions/qc{organ}Dsc0.3",
    f"/work3/kvabo/predictions/qc{organ}Dsc0.2",
    f"/work3/kvabo/predictions/qc{organ}Dsc0.1"
]

# Load full dataset
full_dataset = QCDataset(qc_dirs, dir)

# Convert to DataFrame for filtering
full_df = full_dataset.data

# Get indices for test and train based on case numbers
test_indices = full_df.index[full_df['case'].between(0, 21)].tolist()
train_indices = full_df.index[full_df['case'].between(22, 119)].tolist()
val_indices = full_df.index[full_df['case'].between(120, 140)].tolist()

# Create Subsets
val_dataset = Subset(full_dataset, val_indices)
train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)

# Create DataLoaders
batch_size = 40
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True,persistent_workers=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False,persistent_workers=True)



evalResults = []
evalcsv = Path("Thesis") / f"evalResults{organ}short.csv"
targetClass = 1
with torch.no_grad():
    for batch_idx, (X, Y, case_ids, slice_indices, paths) in enumerate(test_loader):
        logger.info(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
        X = X.to(device)
        Y = Y.to(device)

        outputs = model(X)                          # [B, 2]
        Yhat = torch.sigmoid(outputs)              # [B, 2]

        dsc_error = torch.abs(Yhat[:, 0] - Y[:, 0]) # [B]
        mvsf_error = torch.abs(Yhat[:, 1] - Y[:, 1])
        X = X.detach().cpu().numpy()
        for i in range(X.shape[0]):
            path = paths[i]
            case = case_ids[i].item()
            slice = slice_indices[i].item()
            
            true_dsc = Y[i, 0].item()
            pred_dsc = Yhat[i, 0].item()
            
            true_mvsf = Y[i, 1].item()
            pred_mvsf = Yhat[i, 1].item()
            predseg = X[i,1,:,:].flatten()

            gtSegslice = np.load(dir / f"labels-{case}.npy")[:,:,slice]
            binary_mask = (gtSegslice == targetClass).astype(np.uint8)
    
            # Resize mask
            mask_pil = Image.fromarray(binary_mask * 255)
            mask_resized = TF.resize(mask_pil, (256, 256))
            gtseg = (np.array(mask_resized) / 255.0).flatten()
            
            z = np.sum(predseg)
            y = np.sum(gtseg)
            yz = np.sum(predseg*gtseg)

            evalResults.append({
                "case": case,
                "slice": slice,
                "path": path,
                "preddsc": pred_dsc,
                "predmvsf": pred_mvsf,
                "gtdsc": true_dsc,
                "gtmvsf": true_mvsf,
                "z": z,
                "y": y,
                "yz": yz
            })
        if batch_idx > 10:
            break
        else:
            continue
            

logger.info("Finished evaluation. Saving results to CSV.")

# Save to CSV
with open(evalcsv, 'w', newline='') as f:
    fieldnames = list(evalResults[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(evalResults)

logger.info(f"Saved results to: {evalcsv}")