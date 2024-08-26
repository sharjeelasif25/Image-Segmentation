import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Hyperparameters and paths
IMAGE_HEIGHT = 360  # Adjust based on your model's input size
IMAGE_WIDTH = 640  # Adjust based on your model's input size
INPUT_VIDEO_PATH = "C:/Users/DSU/source/repos/GTA/GTA/video_input/input2.mp4"
OUTPUT_VIDEO_PATH = "C:/Users/DSU/source/repos/GTA/GTA/video_output/transformed_input.mp4"

def get_train_transform():
    return A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            #A.Rotate(limit=35, p=1.0),
            #A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

def apply_transformations_to_frame(frame, transform):
    # Apply transformation
    augmented = transform(image=frame)
    image = augmented["image"].numpy()
    # Convert the tensor to an image format
    image = np.transpose(image, (1, 2, 0))
    image = (image * 255).astype(np.uint8)  # Denormalize to [0, 255]
    return image

def main():
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
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # Get the training transformation
    train_transform = get_train_transform()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply transformations
        transformed_frame = apply_transformations_to_frame(frame, train_transform)

        # Write the transformed frame to the output video
        out.write(transformed_frame)

    # Release everything if job is finished
    cap.release()
    out.release()
    print(f"Output video saved to {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()

