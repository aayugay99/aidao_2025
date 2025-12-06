import torch


def calculate_iou(preds, targets, ignore_index=255, threshold=0.5):
    """
    Считает Macro IoU: (IoU_class_0 + IoU_class_1) / 2.
    Игнорирует пиксели со значением ignore_index (обычно 255).
    """
    # 1. Получаем бинарные предсказания (0 или 1)
    preds_sigmoid = torch.sigmoid(preds)
    preds_bin = preds_sigmoid > threshold  # True, где предсказан класс 1
    
    # 2. Создаем маску валидных пикселей (исключаем 255)
    valid_mask = (targets != ignore_index)
    
    # --- IoU для Класса 1 (Занято / Occupied) ---
    # Intersection: Предсказано 1 И Истина 1 (в валидной зоне)
    tp_1 = (preds_bin & (targets == 1) & valid_mask).sum().float()
    # Union: Предсказано 1 ИЛИ Истина 1 (в валидной зоне)
    union_1 = ((preds_bin | (targets == 1)) & valid_mask).sum().float()
    
    iou_1 = (tp_1 + 1e-6) / (union_1 + 1e-6)
    
    # --- IoU для Класса 0 (Свободно / Free) ---
    # Intersection: Предсказано 0 И Истина 0 (в валидной зоне)
    # ~preds_bin означает "НЕ 1", то есть 0
    tp_0 = ((~preds_bin) & (targets == 0) & valid_mask).sum().float()
    # Union: Предсказано 0 ИЛИ Истина 0 (в валидной зоне)
    union_0 = (((~preds_bin) | (targets == 0)) & valid_mask).sum().float()
    
    iou_0 = (tp_0 + 1e-6) / (union_0 + 1e-6)
    
    # --- Macro Average ---
    macro_iou = (iou_1 + iou_0) / 2
    
    return macro_iou.item()