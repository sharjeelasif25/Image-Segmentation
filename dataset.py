import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.value_to_index = self.create_value_to_index_mapping()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.int64)

        #print(f"Original image shape: {image.shape}, mask shape: {mask.shape}")

        # Ensure that the image and mask shapes match
        if image.shape[:2] != mask.shape:
            #print(f"Resizing mask from shape {mask.shape} to {image.shape[:2]}")
            mask = Image.fromarray(mask.astype(np.uint8)).resize(image.shape[:2][::-1], Image.NEAREST)
            mask = np.array(mask)

        #print(f"Resized image shape: {image.shape}, mask shape: {mask.shape}")

        # Apply the value to index mapping
        mask = np.vectorize(self.value_to_index.get)(mask)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        #print(f"Transformed image shape: {image.shape}, mask shape: {mask.shape}")

        return image, mask

    def create_value_to_index_mapping(self):
        unique_values = [0, 8, 13, 16, 20, 153, 26, 33, 164, 171, 46, 47, 178, 58, 195, 70, 76, 77, 210, 84, 90, 108, 115, 118, 119, 120, 126]
        return {v: i for i, v in enumerate(unique_values)}

    def get_num_classes(self):
        return len(self.value_to_index)
