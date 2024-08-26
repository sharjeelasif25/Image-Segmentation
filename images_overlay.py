import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
from utils import load_checkpoint, get_loaders, apply_color_mapping
import cv2
import numpy as np
import os
from PIL import Image

# Hyperparameters and paths
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # Adjust based on your model's input size
IMAGE_WIDTH = 240  # Adjust based on your model's input size
PIN_MEMORY = True
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"
CHECKPOINT_FILE = "my_checkpoint.pth.tar"
SAVE_IMAGES_FOLDER = "saved_images_overlay"
ALPHA = 0.5  # Transparency factor

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

def overlay_mask_on_image(image, mask, alpha=0.5):
    """
    Overlay the mask on the input image with a specified transparency.
    """
    return cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)

def save_predictions_as_imgs_with_overlay(loader, model, folder="saved_images_overlay/", device="cuda"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    model.eval()
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device=device)
            preds = model(x)
            
            # Convert predictions to class indices
            preds = torch.argmax(preds, dim=1).cpu().numpy()
            x = x.cpu().numpy().transpose(0, 2, 3, 1)  # Move channels to last dimension
            
            for i in range(preds.shape[0]):  # Iterate over batch
                pred_img = preds[i]
                input_img = (x[i] * 255).astype(np.uint8)  # Denormalize and convert to uint8

                # Apply color mapping to the prediction
                pred_img_colored = apply_color_mapping(pred_img)
                
                # Resize colored prediction to the original image size
                pred_img_colored = cv2.resize(pred_img_colored, (input_img.shape[1], input_img.shape[0]))

                # Overlay the mask on the original image
                overlayed_image = overlay_mask_on_image(input_img, pred_img_colored, ALPHA)
                
                # Save the overlayed image
                overlayed_img_pil = Image.fromarray(overlayed_image)
                overlayed_img_pil.save(f"{folder}/overlayed_{idx * preds.shape[0] + i}.png")

    model.train()

def load_and_save_images():
    # Load model
    model = UNET(in_channels=3, out_channels=27).to(DEVICE)  # Adjust out_channels
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Load checkpoint
    load_checkpoint(torch.load(CHECKPOINT_FILE), model)
    
    # Get validation data loader
    _, val_loader = get_loaders(
        train_dir="dummy_dir",  # Placeholder since we only need val_loader
        train_maskdir="dummy_dir",  # Placeholder since we only need val_loader
        val_dir=VAL_IMG_DIR,
        val_maskdir=VAL_MASK_DIR,
        batch_size=BATCH_SIZE,
        train_transform=None,  # Placeholder
        val_transform=val_transforms,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # Save predictions as images with overlay
    save_predictions_as_imgs_with_overlay(val_loader, model, folder=SAVE_IMAGES_FOLDER, device=DEVICE)

if __name__ == "__main__":
    load_and_save_images()

