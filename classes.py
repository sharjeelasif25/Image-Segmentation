import os
import numpy as np
from PIL import Image

def get_unique_values_in_masks(mask_dir):
    unique_values = set()
    for mask_file in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, mask_file)
        mask = np.array(Image.open(mask_path).convert("L"))
        unique_values.update(np.unique(mask))
    return unique_values

mask_dir = 'data/train_masks/'  # Adjust this path to your mask directory
unique_values = get_unique_values_in_masks(mask_dir)
print(f"Unique values in masks: {unique_values}")
print(f"Number of classes: {len(unique_values)}")
print({v: i for i, v in enumerate(unique_values)})
