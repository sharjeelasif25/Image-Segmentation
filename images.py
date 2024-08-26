import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
from utils import load_checkpoint, apply_color_mapping
import cv2
import numpy as np
import os
from PIL import Image

# Hyperparameters and paths
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 160  # Adjust based on your model's input size
IMAGE_WIDTH = 240  # Adjust based on your model's input size
CHECKPOINT_FILE = "my_checkpoint.pth.tar"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"
SAVE_IMAGES_FOLDER = "saved_images"

# Define validation transformations
val_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

def save_predictions_and_gt_masks(model, img_folder=VAL_IMG_DIR, mask_folder=VAL_MASK_DIR, save_folder=SAVE_IMAGES_FOLDER, device="cuda"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    model.eval()
    with torch.no_grad():
        for idx, img_name in enumerate(os.listdir(img_folder)):
            img_path = os.path.join(img_folder, img_name)
            mask_path = os.path.join(mask_folder, img_name)  # Assuming masks have the same name as images

            image = np.array(Image.open(img_path).convert("RGB"))
            gt_mask = np.array(Image.open(mask_path).convert("L"))

            # Apply transformations
            transformed = val_transforms(image=image)
            x = transformed["image"].unsqueeze(0).to(device)  # Add batch dimension

            # Get prediction
            preds = model(x)
            preds = torch.argmax(preds, dim=1).cpu().numpy()
            
            # Convert prediction to color image
            pred_img = preds[0]

            # Apply color mapping to the prediction and GT mask
            pred_img_colored = apply_color_mapping(pred_img)
            gt_mask_colored = apply_color_mapping(gt_mask)

            # Resize colored images to the original image size
            pred_img_colored = cv2.resize(pred_img_colored, (image.shape[1], image.shape[0]))
            gt_mask_colored = cv2.resize(gt_mask_colored, (image.shape[1], image.shape[0]))

            # Save the GT mask and predicted mask images
            gt_mask_pil = Image.fromarray(gt_mask_colored)
            gt_mask_pil.save(f"{save_folder}/gt_mask_{img_name}")

            pred_mask_pil = Image.fromarray(pred_img_colored)
            pred_mask_pil.save(f"{save_folder}/pred_mask_{img_name}")

    model.train()

def load_and_save_images():
    # Load model
    model = UNET(in_channels=3, out_channels=27).to(DEVICE)  # Adjust out_channels
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Load checkpoint
    load_checkpoint(torch.load(CHECKPOINT_FILE), model)
    
    # Save predictions and GT masks
    save_predictions_and_gt_masks(model, img_folder=VAL_IMG_DIR, mask_folder=VAL_MASK_DIR, save_folder=SAVE_IMAGES_FOLDER, device=DEVICE)

if __name__ == "__main__":
    load_and_save_images()

