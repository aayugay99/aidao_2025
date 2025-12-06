import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import v2

from PIL import Image

from pathlib import Path


CAMERA_NAMES = [
    "/camera/inner/frontal/middle",
    "/camera/inner/frontal/far",
    "/side/left/forward",
    "/side/right/forward",
]

INTRINSICS_NAMES = [
    "/camera/inner/frontal/middle/intrinsic_params",
    "/camera/inner/frontal/far/intrinsic_params",
    "/side/left/forward/intrinsic_params",
    "/side/right/forward/intrinsic_params",
]

CAR2CAM_NAMES = [
    "/camera/inner/frontal/middle/car_to_cam",
    "/camera/inner/frontal/far/car_to_cam",
    "/side/left/forward/car_to_cam",
    "/side/right/forward/car_to_cam",
]

GRIDS_NAMES = [
    "gt_occupancy_grid",
]

class BaseDataset(Dataset):
    def __init__(self, data_dir: Path, mode: str = "train"):
        self.mode = mode
        self.transform = v2.Compose([
            v2.PILToTensor(),
            v2.Resize((256, 512)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.data_dir = data_dir
        self.info = pd.read_csv(data_dir / "info.csv", index_col=0)
        self.images_paths = []
        self.intrinsics_paths = []
        self.car2cam_paths = []
        if self.mode != "test":
            self.static_grids_paths = []

        for _, row in self.info.iterrows():
            self.images_paths.append([row[name] for name in CAMERA_NAMES])                
            self.intrinsics_paths.append([row[name] for name in INTRINSICS_NAMES])
            self.car2cam_paths.append([row[name] for name in CAR2CAM_NAMES])
            if self.mode != "test":
                self.static_grids_paths.append([row[name] for name in GRIDS_NAMES])

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        images = torch.stack([self.transform(Image.open(self.data_dir.parent / img_path)) for img_path in self.images_paths[idx]], dim=0)
        intrinsics = np.stack([np.load(self.data_dir.parent / intr_path) for intr_path in self.intrinsics_paths[idx]], axis=0)
        car2cams = np.stack([np.load(self.data_dir.parent / car2cam_path) for car2cam_path in self.car2cam_paths[idx]], axis=0)

        if self.mode != "test":
            static_grids = np.load(self.data_dir.parent / self.static_grids_paths[idx][0]) 
            return images, intrinsics, car2cams, static_grids

        return images, intrinsics, car2cams
