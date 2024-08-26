# utils.py

import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir=None,
    train_maskdir=None,
    val_dir=None,
    val_maskdir=None,
    batch_size=16,
    train_transform=None,
    val_transform=None,
    num_workers=4,
    pin_memory=True,
):
    train_loader = None
    if train_dir and train_maskdir and os.path.exists(train_dir) and os.path.exists(train_maskdir):
        train_ds = CarvanaDataset(
            image_dir=train_dir,
            mask_dir=train_maskdir,
            transform=train_transform,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
        )

    val_loader = None
    if val_dir and val_maskdir and os.path.exists(val_dir) and os.path.exists(val_maskdir):
        val_ds = CarvanaDataset(
            image_dir=val_dir,
            mask_dir=val_maskdir,
            transform=val_transform,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            num_correct += (preds == y).sum().item()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum().item()) / (
                (preds + y).sum().item() + 1e-8
            )

    accuracy = num_correct / num_pixels * 100
    print(f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}%")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    
    return accuracy  # Return accuracy

def apply_color_mapping(mask):
    COLOR_MAPPING = {
        0: (0, 0, 0), 1: (128, 0, 0), 2: (0, 128, 0), 3: (128, 128, 0),
        4: (0, 0, 128), 5: (128, 0, 128), 6: (0, 128, 128), 7: (128, 128, 128),
        8: (64, 0, 0), 9: (192, 0, 0), 10: (64, 128, 0), 11: (192, 128, 0),
        12: (64, 0, 128), 13: (192, 0, 128), 14: (64, 128, 128), 15: (192, 128, 128),
        16: (0, 64, 0), 17: (128, 64, 0), 18: (0, 192, 0), 19: (128, 192, 0),
        20: (0, 64, 128), 21: (128, 64, 128), 22: (0, 192, 128), 23: (128, 192, 128),
        24: (64, 64, 0), 25: (192, 64, 0), 26: (64, 192, 0)
    }
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for class_idx, color in COLOR_MAPPING.items():
        color_mask[mask == class_idx] = color
    return color_mask

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    model.eval()
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device=device)
            preds = model(x)
            
            # Convert predictions to class indices
            preds = torch.argmax(preds, dim=1)  # Shape: [batch_size, height, width]
            preds = preds.cpu().numpy()
            
            # Save predictions
            for i in range(preds.shape[0]):  # Iterate over batch
                pred_img = preds[i]
                
                # Apply color mapping
                pred_img_colored = apply_color_mapping(pred_img)
                
                # Convert to PIL image
                pred_img_pil = Image.fromarray(pred_img_colored)
                pred_img_pil.save(f"{folder}/pred_{idx * preds.shape[0] + i}.png")

            # Save ground truth
            y = y.to(device=device)
            y = y.cpu().numpy()
            
            for i in range(y.shape[0]):  # Iterate over batch
                gt_img = y[i]
                
                # Apply color mapping
                gt_img_colored = apply_color_mapping(gt_img)
                
                # Convert to PIL image
                gt_img_pil = Image.fromarray(gt_img_colored)
                gt_img_pil.save(f"{folder}/gt_{idx * y.shape[0] + i}.png")

    model.train()
