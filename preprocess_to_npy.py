import nibabel as nib
import numpy as np
from pathlib import Path
import argparse

def interval_mapping(image, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

def convert_nifti_to_npy(data_dir: Path, case_numbers: list, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for case_number in case_numbers:
        vol_path = data_dir / f"volume-{case_number}.nii.gz"
        label_path = data_dir / f"labels-{case_number}.nii.gz"

        # Load NIfTI volumes
        volume = np.array(nib.load(str(vol_path)).get_fdata(), dtype=np.float32)
        label = np.array(nib.load(str(label_path)).get_fdata(), dtype=np.float16)

        # Normalize
        # clip 
        volume = np.clip(volume, -1000, 1000)
        
        # convert to 0-1
        volume = interval_mapping(volume, -1000, 1000, 0, 1)

        # Save as .npy files

        np.save(output_dir / f"volume-{case_number}.npy", volume)
        np.save(output_dir / f"labels-{case_number}.npy", label)

        print(f"Saved volume-{case_number}.npy and labels-{case_number}.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to raw NIfTI .nii.gz files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save preprocessed .npy files")
    parser.add_argument("--cases", nargs="+", required=True, help="List of case numbers to convert")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    case_numbers = list(map(int, args.cases))

    convert_nifti_to_npy(data_dir, case_numbers, output_dir)
