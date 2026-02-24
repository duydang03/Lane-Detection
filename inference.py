import os
import argparse
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from models import build_model


# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Transform (same as val)
# =========================
def get_transform(img_size=512):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# =========================
# Load Model
# =========================
def load_model(model_name, checkpoint_path, num_classes=1):

    model = build_model(model_name, num_classes=num_classes)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    print(f"✓ Loaded model from {checkpoint_path}")
    return model


# =========================
# Predict single image
# =========================
def predict_image(model, image_path, transform, threshold=0.5):

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    transformed = transform(image=original_image)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        pred = (output > threshold).float()

    mask = pred.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)

    return original_image, mask


# =========================
# Overlay mask
# =========================
def overlay_mask(image, mask, alpha=0.4):

    color_mask = np.zeros_like(image)
    color_mask[:, :, 1] = mask  # Green channel

    overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    return overlay


# =========================
# Predict folder
# =========================
def predict_folder(model, input_dir, output_dir, img_size=512):

    os.makedirs(output_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, "masks")
    overlay_dir = os.path.join(output_dir, "overlays")

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    transform = get_transform(img_size)

    image_paths = [os.path.join(input_dir, f)
                   for f in os.listdir(input_dir)
                   if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    print(f"Found {len(image_paths)} images")

    for img_path in tqdm(image_paths):

        filename = os.path.basename(img_path)

        original_image, mask = predict_image(model, img_path, transform)

        overlay = overlay_mask(original_image, mask)

        # Convert back to BGR for saving
        original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(mask_dir, filename), mask)
        cv2.imwrite(os.path.join(overlay_dir, filename), overlay_bgr)

    print("✓ Inference completed!")


# =========================
# Main
# =========================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True,
                        choices=["espnet", "deeplab", "twin", "unet"])

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="inference_results")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=1)

    args = parser.parse_args()

    model = load_model(args.model, args.checkpoint, args.num_classes)

    predict_folder(
        model=model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        img_size=args.img_size
    )


if __name__ == "__main__":
    main()