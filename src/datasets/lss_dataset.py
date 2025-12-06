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

class BEVDataset(Dataset):
    def __init__(self, data_dir: Path, mode: str = "train", image_size=(256, 512)):
        """
        data_dir: Путь к папке конкретного сплита (например, .../dataset_train)
        mode: 'train', 'val' (возвращают GT) или 'test' (не возвращает GT)
        """
        self.mode = mode
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        # Читаем CSV. Ожидается, что info.csv лежит внутри data_dir
        csv_path = self.data_dir / "info.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
            
        self.info = pd.read_csv(csv_path, index_col=0)
        
        # Корневая папка для путей из CSV.
        # Если в CSV пути вида "autonomy_yandex_dataset_train/images/...", 
        # а data_dir = ".../autonomy_yandex_dataset_train", 
        # то нам нужно подняться на уровень выше.
        self.data_root = self.data_dir.parent 
        
        # Препроцессинг
        self.transform = v2.Compose([
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(self.image_size),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        # Кэшируем список семплов для быстрого доступа по индексу
        self.samples = []
        for idx, row in self.info.iterrows():
            sample = {
                "images": [row[name] for name in CAMERA_NAMES],
                "intrinsics": [row[name] for name in INTRINSICS_NAMES],
                "car2cam": [row[name] for name in CAR2CAM_NAMES],
            }
            # GT нужен только для train и val
            if self.mode in ["train", "val"]:
                sample["gt_grid"] = row["gt_occupancy_grid"]
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        imgs = []
        intrinsics = []
        car2cams = []
        
        for i in range(4):
            # 1. Загрузка изображения
            # Используем data_root, так как пути в CSV относительные
            img_path = self.data_root / sample["images"][i]
            pil_img = Image.open(img_path)
            orig_w, orig_h = pil_img.size
            
            # 2. Трансформация
            img_tensor = self.transform(pil_img)
            imgs.append(img_tensor)
            
            # 3. Загрузка Intrinsics
            intr_path = self.data_root / sample["intrinsics"][i]
            intr = np.load(intr_path) 
            
            # --- Fix 3x4 to 3x3 ---
            if intr.shape == (3, 4):
                intr = intr[:3, :3]
            
            # Масштабирование матрицы камеры под ресайз картинки
            scale_x = self.image_size[1] / orig_w
            scale_y = self.image_size[0] / orig_h
            intr[0, 0] *= scale_x # fx
            intr[0, 2] *= scale_x # cx
            intr[1, 1] *= scale_y # fy
            intr[1, 2] *= scale_y # cy
            
            intrinsics.append(torch.from_numpy(intr).float())
            
            # 4. Загрузка Extrinsics
            c2c_path = self.data_root / sample["car2cam"][i]
            c2c = np.load(c2c_path)
            car2cams.append(torch.from_numpy(c2c).float())

        imgs = torch.stack(imgs)
        intrinsics = torch.stack(intrinsics)
        car2cams = torch.stack(car2cams)
        
        # Инверсия: нам нужно camera -> car (для LSS)
        cam2cars = torch.inverse(car2cams)

        # Возврат данных в зависимости от режима
        if self.mode in ["train", "val"]:
            gt_path = self.data_root / sample["gt_grid"]
            gt_grid = np.load(gt_path).astype(np.int64)
            if len(gt_grid.shape) == 2:
                gt_grid = gt_grid[None, ...] # (1, H, W)
            return imgs, intrinsics, cam2cars, torch.from_numpy(gt_grid)
        
        # Для теста GT не возвращаем
        return imgs, intrinsics, cam2cars
