import torch

from datetime import datetime

from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np

from .metrics import calculate_iou

def create_dataloaders(train_dir, val_dir, batch_size=8, num_workers=8):
    train_dataset = BaseDataset(data_dir=Path(train_dir), mode="train")
    val_dataset = BaseDataset(data_dir=Path(val_dir), mode="val")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    losses = []
    ious = []

    pbar = tqdm(train_loader)
    for images, intrinsics, car2cams, gt in pbar:
        images = images.to(device)
        intrinsics = intrinsics.to(device)
        car2cams = car2cams.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()

        preds = model(images, intrinsics, car2cams)
        loss_mask = gt != 255
        loss = criterion(preds[loss_mask], gt[loss_mask].float())
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        iou = calculate_iou(preds, gt)
        ious.append(iou)
        pbar.set_postfix({"Loss": loss.item(), "Train IOU": iou})

    return np.mean(losses), np.mean(ious)


def validate(model, val_loader, criterion, device):
    model.eval()
    losses = []
    ious = []

    with torch.no_grad():
        for images, intrinsics, car2cams, gt in tqdm(val_loader):
            images = images.to(device)
            intrinsics = intrinsics.to(device)
            car2cams = car2cams.to(device)
            gt = gt.to(device)

            preds = model(images, intrinsics, car2cams)
            loss_mask = gt != 255
            loss = criterion(preds[loss_mask], gt[loss_mask].float())

            losses.append(loss.item())
            ious.append(calculate_iou(preds, gt))

    return np.mean(losses), np.mean(ious)


def create_run_dir(base_dir="checkpoints"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_checkpoint(model, epoch, val_iou, ckpt_dir="checkpoints"):
    Path(ckpt_dir).mkdir(exist_ok=True)
    ckpt_path = Path(ckpt_dir) / f"epoch{epoch}_iou{val_iou:.4f}.pth"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def train_model(model, optimizer, criterion, device, train_dir, val_dir,
                epochs=100, batch_size=8, num_workers=8, ckpt_dir="checkpoints"):

    run_dir = create_run_dir(ckpt_dir)

    train_loader, val_loader = create_dataloaders(train_dir, val_dir,
                                                  batch_size, num_workers)
    best_val_iou = 0

    for epoch in range(epochs):
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_iou = validate(model, val_loader, criterion, device)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_checkpoint(model, epoch, val_iou, run_dir)

        print(f"Epoch {epoch}: Train IOU = {train_iou:.4f}, Val IOU = {val_iou:.4f}")

    print(f"Best validation IOU: {best_val_iou:.4f}")
