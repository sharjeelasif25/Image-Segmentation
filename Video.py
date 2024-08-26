import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
from utils import load_checkpoint, apply_color_mapping
import numpy as np
from PIL import Image
import os

# Hyperparameters and paths
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 480  # Adjust based on model's input size
IMAGE_WIDTH = 854  # Adjust based on model's input size
CHECKPOINT_FILE = "my_checkpoint.pth.tar"
INPUT_VIDEO_PATH = "C:/Users/DSU/source/repos/GTA/GTA/video_input/input3.mp4"
OUTPUT_VIDEO_PATH = "C:/Users/DSU/source/repos/GTA/GTA/video_output/output 5.mp4"

# Define transformation
val_transform = A.Compose(
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

def process_frame(frame, model, transform, device):
    # Apply transformation
    augmented = transform(image=frame)
    image = augmented["image"].unsqueeze(0).to(device)

    # Get prediction from model
    model.eval()
    with torch.no_grad():
        preds = model(image)
        preds = torch.argmax(preds, dim=1).cpu().numpy()
    
    # Convert prediction to color image
    pred_img_colored = apply_color_mapping(preds[0])
    return pred_img_colored

def temporal_smoothing(frames, alpha=0.6):
    smoothed_frame = frames[0].astype(np.float32)
    for i in range(1, len(frames)):
        smoothed_frame = alpha * smoothed_frame + (1 - alpha) * frames[i].astype(np.float32)
    return smoothed_frame.astype(np.uint8)

def main():
    # Load model
    model = UNET(in_channels=3, out_channels=27).to(DEVICE)  # Adjust out_channels
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Load checkpoint
    load_checkpoint(torch.load(CHECKPOINT_FILE), model)
    
    print(f"Input video path: {INPUT_VIDEO_PATH}")
    print(f"File exists: {os.path.exists(INPUT_VIDEO_PATH)}")
    
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

    processed_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to the model's input size
        frame_resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # Process frame
        output_frame = process_frame(frame_resized, model, val_transform, DEVICE)
        
        # Resize output frame back to original size
        output_frame_resized = cv2.resize(output_frame, (frame_width, frame_height))
        processed_frames.append(output_frame_resized)
        
        # Apply temporal smoothing every 10 frames
        if len(processed_frames) == 10:
            smoothed_frame = temporal_smoothing(processed_frames)
            out.write(smoothed_frame)
            processed_frames.pop(0)

    # Write any remaining frames
    for frame in processed_frames:
        out.write(frame)

    # Release everything if job is finished
    cap.release()
    out.release()
    print(f"Output video saved to {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()
