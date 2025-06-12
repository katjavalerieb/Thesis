import numpy as np
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

def generate_slice_map(data_dir, case_numbers, output_path):
    data_dir = Path(data_dir)
    slice_data = []

    organClass = 5

    for case_number in case_numbers:
        label_path = data_dir / f"labels-{case_number}.npy"
        label = np.load(label_path)  # shape: (H, W, D)

        # Vectorized mask: (H, W, D) -> (D,)
        has_organ = np.any(label == organClass, axis=(0, 1))

        for i, present in enumerate(has_organ[:-1]):  # Skip last slice
            if present:
                slice_data.append((case_number, i))

        print(case_number)

    # Save to .pkl
    with open(output_path, 'wb') as f:
        pickle.dump(slice_data, f)

    print(f"Saved {len(slice_data)} kidney-containing slices to {output_path}")


# Example usage
if __name__ == "__main__":
    case_numbers = list(range(140))  # Adjust to your dataset
    generate_slice_map(
        data_dir=Path("/work3/kvabo/PKGCTORG/newCT"),
        case_numbers=case_numbers,
        output_path="boneSlices.pkl"
    )
