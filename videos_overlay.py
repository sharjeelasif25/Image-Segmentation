import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
from utils import load_checkpoint, apply_color_mapping
import cv2
import numpy as np
from PIL import Image

# Hyperparameters and paths
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 360  # Adjust based on your model's input size
IMAGE_WIDTH = 640  # Adjust based on your model's input size
CHECKPOINT_FILE = "my_checkpoint.pth.tar"
INPUT_VIDEO_PATH = "C:/Users/DSU/source/repos/GTA/GTA/video_input/input2.mp4"
OUTPUT_VIDEO_PATH = "C:/Users/DSU/source/repos/GTA/GTA/video_output/output 5 (Overlay).mp4"
ALPHA = 0.5  # Transparency factor

# Define transformation
val_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

def overlay_mask_on_image(image, mask, alpha=0.5):
    """
    Overlay the mask on the input image with a specified transparency.
    """
    return cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)

def process_frame(frame, model, transform, device):
    # Apply transformations
    augmented = transform(image=frame)
    image = augmented["image"].unsqueeze(0).to(device)

    # Get prediction from model
    model.eval()
    with torch.no_grad():
        preds = model(image)
        preds = torch.argmax(preds, dim=1).cpu().numpy()

    # Convert prediction to color image
    pred_img_colored = apply_color_mapping(preds[0])

    # Resize colored prediction to the original image size
    pred_img_colored = cv2.resize(pred_img_colored, (frame.shape[1], frame.shape[0]))

    # Overlay the mask on the original image
    overlayed_image = overlay_mask_on_image(frame, pred_img_colored, ALPHA)
    
    return overlayed_image

def main():
    # Load model
    model = UNET(in_channels=3, out_channels=27).to(DEVICE)  # Adjust out_channels
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Load checkpoint
    load_checkpoint(torch.load(CHECKPOINT_FILE), model)
    
    # Open input video
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open input video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        output_frame = process_frame(frame, model, val_transforms, DEVICE)

        # Write the frame to the output video
        out.write(output_frame)

    # Release everything if job is finished
    cap.release()
    out.release()
    print(f"Output video saved to {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()
