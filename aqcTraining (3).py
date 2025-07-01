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
from functools import lru_cache

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dir = Path("/work3/kvabo/PKGCTORG/newCT")

parser = argparse.ArgumentParser(description='Automatic Quality Control Model Training')
parser.add_argument('--organ', type=str, required=True, help='Liver, Bone or Kidney')

args = parser.parse_args()

organ = args.organ

logging.basicConfig(
    filename=f'aqc{organ}V4.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

##################### MODEL #####################

import torch
import torch.nn as nn

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

def init_he(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


model = QCUNet()
model_path = f"Thesis/aqcModel{organ}V3.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

#model.apply(init_he)
model.to(device)
##################### DATA LOADING #####################


class QCDataset(Dataset):
    def __init__(self, qc_dirs, ct_dir):
        self.ct_dir = Path(ct_dir)

        self.data = []

        # Load and combine all resultsUpdate.csv files
        for qc_dir in qc_dirs:
            qc_path = Path(qc_dir)
            #df = pd.read_csv(qc_path / "resultsUpdate.csv")
            df = pd.read_csv(qc_path / "results.csv")
            df["qc_dir"] = qc_path  # Track where each sample comes from
            self.data.append(df)

        self.data = pd.concat(self.data, ignore_index=True)

        # Sort the data so we consistently run through the same stuff
        # and improve the LRU cache
        self.data = self.data.sort_values(by=["qc_dir", "case", "slice"])

    def __len__(self):
        return len(self.data)

    def load_seq_file(self, qc_dir, case_number, slice_idx):
        # Load segmentation
        seg_path = qc_dir / f"{case_number}_slice_{slice_idx}_pred.npy"
        seg = np.load(seg_path).astype(np.float32)
        seg_tensor = torch.from_numpy(seg).unsqueeze(0)  # shape: [1, 256, 256]
        return seg_tensor

    # The ct file seems huge (volume), reading slices is bad.
    # So likely maxsize shouldn't be too big, start with 1 and see how it performs.
    @lru_cache(maxsize=1)
    def load_ct_file(self, case_number):
        # Load and resize CT slice
        ct_path = self.ct_dir / f"volume-{case_number}.npy"
        ct = np.load(ct_path).astype(np.float32)
        return ct


    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        case_number = int(row['case'])
        slice_idx = int(row['slice'])
        qc_dir = Path(row['qc_dir'])

        # Target vector: [DSC, MVSF]
        Y = torch.tensor([row['dsc'], row['mvsf']], dtype=torch.float32)

        seg_tensor = self.load_seq_file(qc_dir, case_number, slice_idx)

        # Load and resize CT slice
        ct = self.load_ct_file(case_number)[..., slice_idx]
        ct_tensor = torch.from_numpy(ct).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]
        ct_resized = F.interpolate(ct_tensor, size=(256, 256), mode='bilinear', align_corners=False)
        ct_resized = ct_resized.squeeze(0)  # shape: [1, 256, 256]

        # Stack CT and segmentation
        X = torch.cat([ct_resized, seg_tensor], dim=0)  # shape: [2, 256, 256]

        return X, Y, case_number, slice_idx,


qc_dirs = [
    f"/work3/kvabo/predictions/qc{organ}Dsc1.0",
    f"/work3/kvabo/predictions/qc{organ}Dsc0.9",
    f"/work3/kvabo/predictions/qc{organ}Dsc0.8",
    f"/work3/kvabo/predictions/qc{organ}Dsc0.7",
    f"/work3/kvabo/predictions/qc{organ}Dsc0.6",
    f"/work3/kvabo/predictions/qc{organ}Dsc0.5",
    f"/work3/kvabo/predictions/qc{organ}Dsc0.4",
    f"/work3/kvabo/predictions/qc{organ}Dsc0.3",
    f"/work3/kvabo/predictions/qc{organ}Dsc0.2"
    ,f"/work3/kvabo/predictions/qc{organ}Dsc0.1"
]

# Load full dataset
full_dataset = QCDataset(qc_dirs, dir)

# Convert to DataFrame for filtering
full_df = full_dataset.data

# Get indices for test and train based on case numbers
test_indices = full_df.index[full_df['case'].between(0, 20)].tolist()
train_indices = full_df.index[full_df['case'].between(21, 120)].tolist()
val_indices = full_df.index[full_df['case'].between(121, 139)].tolist()

# Create Subsets
val_dataset = Subset(full_dataset, val_indices)
train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)

# Create DataLoaders
batch_size = 40
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False,persistent_workers=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False,persistent_workers=True)


##################### LOSS & OPTIMIZER #####################
maeLoss = nn.L1Loss()
opt = torch.optim.Adam(model.parameters(), lr=1e-4) 
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.7)
num_epochs = 10


##################### TRAINING #####################

# Training logger
train_logger = logging.getLogger("train_logger")
train_handler = logging.FileHandler(f"aqc{organ}_train_V4.log", mode='w')
train_handler.setFormatter(logging.Formatter('%(message)s'))
train_logger.setLevel(logging.INFO)
train_logger.addHandler(train_handler)
train_logger.propagate = False  # Prevents duplicate logs if root logger is used

# Validation logger
val_logger = logging.getLogger("val_logger")
val_handler = logging.FileHandler(f"aqc{organ}_val_V4.log", mode='w')
val_handler.setFormatter(logging.Formatter('%(message)s'))
val_logger.setLevel(logging.INFO)
val_logger.addHandler(val_handler)
val_logger.propagate = False


train_loss_path = Path(f"Thesis/train_batch_losses_{organ}_V4.csv")
val_loss_path = Path(f"Thesis/val_batch_losses_{organ}_V4.csv")

# Open both CSV files
with open(train_loss_path, mode='w', newline='') as train_file, \
     open(val_loss_path, mode='w', newline='') as val_file:

    train_writer = csv.writer(train_file)
    val_writer = csv.writer(val_file)

    # Write headers
    train_writer.writerow(["Epoch", "Loss"])
    val_writer.writerow(["Epoch", "Loss"])


    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs} - Training Started")
        
        model.eval()
        valLoss = 0.0
        for X, Y, case_ids, slice_indices in val_loader:
            X = X.to(device)
            Y = Y.to(device)
            outputs = model(X)  # shape: [batch_size, 2]
            Yhat = torch.sigmoid(outputs).squeeze(0)  # shape: [2]
            loss  = maeLoss(Yhat, Y)
            valLoss += loss.item()
            # Log
            logger.info(f"Validation: Epoch {epoch+1:02d} Loss: {loss.item():.4f}")
            val_writer.writerow([epoch+1, loss.item()])
            val_logger.info(loss.item())
        
        model.train()
        trainLoss = 0.0
        
        for X, Y, case_ids, slice_indices in train_loader:
            X = X.to(device)
            Y = Y.to(device)
        
            opt.zero_grad()
            outputs = model(X)  # shape: [batch_size, 2]
            Yhat = torch.sigmoid(outputs).squeeze(0)  # shape: [2]
        
            loss  = maeLoss(Yhat, Y)
            loss.backward()
            opt.step()
            trainLoss += loss.item()
        
            # Log
            logger.info(f"Training: Epoch {epoch+1:02d} Loss: {loss.item():.4f}")
            
            torch.save(model.state_dict(), Path(f"Thesis/aqcModel{organ}V4.pth"))
            train_writer.writerow([epoch + 1, loss.item()])
            train_logger.info(loss.item())
    
    
    
        avgTrainLoss = trainLoss / len(train_loader)
        avgValLoss = valLoss / len(val_loader)
        #writer.writerow([epoch, avgTrainLoss, avgValLoss])
        
        scheduler.step()
        logger.info(f"Epoch {epoch} | Train Loss: {avgTrainLoss:.4f} | Val Loss: {avgValLoss:.4f}")
        #torch.save(model.state_dict(), Path(f"/work3/kvabo/predictions/aqcModel{organ}.pth"))
        break
    
logger.info("Training completed.")