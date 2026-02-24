import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import albumentations as A
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import segmentation_models_pytorch as smp
from glob import glob
from Utils import get_matching_image_mask_paths
from torchvision import transforms
from ESPNet_custom import ESPNet
from TwinLiteNet import TwinLiteNet
from torch.amp import autocast, GradScaler
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader,random_split,Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models import build_model
from losses.combo_loss import ComboLoss
from metrics.metrics import calculate_detailed_metrics, EarlyStopping
from data.lane_dataset import LaneSegDataset


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#Validate
def validate(model, val_loader, criterion, device):

    model.eval()
    val_loss = 0.0

    dice_total = 0.0
    iou_total = 0.0
    precision_total = 0.0
    recall_total = 0.0
    f1_total = 0.0

    with torch.no_grad():
        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device).float()

            outputs = model(images)

            if outputs.shape != masks.shape:
                outputs = F.interpolate(
                    outputs,
                    size=masks.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )

            loss = criterion(outputs, masks)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            metrics = calculate_detailed_metrics(preds, masks)

            dice_total += metrics["dice"]
            iou_total += metrics["iou"]
            precision_total += metrics["precision"]
            recall_total += metrics["recall"]
            f1_total += metrics["f1"]

    num_batches = len(val_loader)

    return {
        "loss": val_loss / num_batches,
        "dice": dice_total / num_batches,
        "iou": iou_total / num_batches,
        "precision": precision_total / num_batches,
        "recall": recall_total / num_batches,
        "f1": f1_total / num_batches,
    }


def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True,
                        choices=["espnet", "deeplab", "twin", "unet"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--train_img_dir", type=str, required=True)
    parser.add_argument("--train_mask_dir", type=str, required=True)
    parser.add_argument("--val_img_dir", type=str, required=True)
    parser.add_argument("--val_mask_dir", type=str, required=True)

    args = parser.parse_args()
    
    #Transform
    transform = A.Compose([
        A.Resize(512, 512),
        A.RandomBrightnessContrast(p=0.3),
        A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-10, 10), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ElasticTransform(p=0.2),
        A.GridDistortion(p=0.2),
        A.CoarseDropout(max_holes=8, max_height=64, max_width=64, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], is_check_shapes=False)
    def transform_fn(image, mask):
        transformed = transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']

    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ],is_check_shapes=False)
    def val_transform_fn(image, mask):
        transformed = val_transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(42)

    train_img_paths, train_mask_paths = get_matching_image_mask_paths(args.train_img_dir, args.train_mask_dir)
    val_img_paths, val_mask_paths = get_matching_image_mask_paths(args.val_img_dir, args.val_mask_dir)


    #Tạo DataLoader
    torch.backends.cudnn.benchmark = True
    val_dataset = LaneSegDataset(val_img_paths, val_mask_paths, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,  num_workers=2,pin_memory=True)

    train_dataset = LaneSegDataset(train_img_paths, train_mask_paths, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True , num_workers=2,pin_memory=True)

    model = build_model(args.model, num_classes=1)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=7,factor=0.3,min_lr=1e-7,verbose=True)
    criterion = ComboLoss(alpha=0.5)
    early_stopping = EarlyStopping(
        patience=10,
        min_delta=0.001
    )

    num_epochs = args.epochs
    start_epoch = 0
    end_epoch = start_epoch + num_epochs

    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    train_loss_history = []
    val_loss_history = []
    dice_history = []
    iou_history = []
    precision_history = []
    recall_history = []
    f1_history = []
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(start_epoch, end_epoch):

        # =========================
        # TRAIN
        # =========================
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                outputs = F.interpolate(
                    outputs,
                    size=masks.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # =========================
        # VALIDATE
        # =========================
        val_metrics = validate(model, val_loader, criterion, device)

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(val_metrics["loss"])
        dice_history.append(val_metrics["dice"])
        iou_history.append(val_metrics["iou"])
        precision_history.append(val_metrics["precision"])
        recall_history.append(val_metrics["recall"])
        f1_history.append(val_metrics["f1"])

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch [{epoch+1}/{end_epoch}]")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val   Loss: {val_metrics['loss']:.4f}")
        print(f"Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f} | F1: {val_metrics['f1']:.4f}")
        print(f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
        print(f"LR: {current_lr:.6f}")

        # =========================
        # SAVE BEST MODEL
        # =========================
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_dir = "checkpoints"
            checkpoint_path = os.path.join(
                save_dir,
                f"{args.model}_best.pth"
            )

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, checkpoint_path)

            print(f"✓ Best model saved at {checkpoint_path}")

        # =========================
        # LR Scheduler
        # =========================
        if scheduler is not None:
            scheduler.step(val_metrics["loss"])

        # =========================
        # Early Stopping
        # =========================
        if early_stopping is not None:
            if early_stopping(val_metrics["loss"], model):
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best validation loss: {early_stopping.best_loss:.4f}")

                if early_stopping.restore_best_weights:
                    early_stopping.load_best_weights(model)
                    print("Restored best model weights")

                break

    print("\nTraining completed!")

if __name__ == "__main__":
    main()
