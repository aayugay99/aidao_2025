import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights


class CamEncoder(nn.Module):
    """
    Извлекает признаки и предсказывает распределение глубины для изображения.
    Backbone: Swin Transformer (swin_t).
    """
    def __init__(self, D, C, downsample=32):
        super(CamEncoder, self).__init__()
        self.D = D  # depth bins
        self.C = C  # context channels

        # --- SWIN BACKBONE ---
        # swin_t: patch_size=4, итоговый stride = 32 (как у ResNet50 layer4)
        weights = Swin_T_Weights.IMAGENET1K_V1
        self.backbone = swin_t(weights=weights)

        # Кол-во каналов последнего уровня (embed_dim * 2 ** (len(depths)-1))
        # Это то же число, что и in_features у классификационной головы.
        in_channels = self.backbone.head.in_features  # для swin_t = 768

        # --- HEAD ДЛЯ ГЛУБИНЫ+КОНТЕКСТА ---
        # Вход: (B, in_channels, H/32, W/32)
        # Выход: (B, D + C, H/32, W/32)
        self.depth_net = nn.Conv2d(in_channels, self.D + self.C, kernel_size=1)

    def get_depth_dist(self, x):
        # x: (B, D + C, H, W)
        return x[:, :self.D].softmax(dim=1)

    def get_context(self, x):
        return x[:, self.D:]

    def forward(self, x):
        # x: (B*N, 3, H, W)

        # Swin сам делает patch embedding и все стадии ↓ разрешения.
        # features: (B*N, H/32, W/32, C)  — ВАЖНО: channel last!
        x = self.backbone.features(x)          # (B, H', W', C)
        x = self.backbone.norm(x)              # (B, H', W', C)

        # Переводим в BCHW (channel first)
        # Можно использовать встроенный permute-модуль из модели:
        x = self.backbone.permute(x)           # (B, C, H', W')

        # Дальше всё как раньше
        x = self.depth_net(x)                  # (B, D + C, H', W')
        depth = self.get_depth_dist(x)         # (B, D, H', W')
        context = self.get_context(x)          # (B, C, H', W')

        return depth, context
    

class ViewTransformer(nn.Module):
    """
    Переводит 2D признаки в 3D облако точек и проецирует их в BEV сетку.
    """
    def __init__(self, grid_conf, data_conf):
        super(ViewTransformer, self).__init__()
        self.grid_conf = grid_conf
        self.data_conf = data_conf
        
        dx, bx, nx = self.gen_dx_bx(self.grid_conf['x_bound'], self.grid_conf['y_bound'], self.grid_conf['z_bound'])
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.D = int((data_conf['d_bound'][1] - data_conf['d_bound'][0]) / data_conf['d_bound'][2])
        self.frustum = self.create_frustum()

    def gen_dx_bx(self, xbound, ybound, zbound):
        dx = torch.Tensor([0.8, 0.8, 20.0]) 
        bx = torch.Tensor([xbound[0] + 0.4, ybound[0] + 0.4, zbound[0]])
        nx = torch.LongTensor([(xbound[1] - xbound[0]) / 0.8, (ybound[1] - ybound[0]) / 0.8, 1])
        return dx, bx, nx

    def create_frustum(self):
        feat_h, feat_w = 8, 16 
        ds = torch.arange(*self.data_conf['d_bound'], dtype=torch.float).view(-1, 1, 1).expand(-1, feat_h, feat_w)
        D, H, W = ds.shape
        xs = torch.linspace(0, self.data_conf['image_size'][1] - 1, W).view(1, 1, W).expand(D, H, W)
        ys = torch.linspace(0, self.data_conf['image_size'][0] - 1, H).view(1, H, 1).expand(D, H, W)
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrinsics):
        B, N, _ = trans.shape
        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combined_transform = torch.inverse(intrinsics)
        points = combined_transform.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points = rots.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        return points

    def voxel_pooling(self, geom_feats, x):
        """
        Складывает фичи, попавшие в один воксель.
        """
        B, N, D, H, W, _ = geom_feats.shape
        
        # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
        # Используем reshape вместо view, так как данные могут быть не contiguous
        # Либо явно вызываем contiguous()
        geom_feats = geom_feats.reshape(B, -1, 3) 
        feats = x.reshape(B, -1, x.shape[-1])
        # -------------------------
        
        long_coords = ((geom_feats - self.bx.to(geom_feats.device)) / self.dx.to(geom_feats.device)).long()
        
        valid = (long_coords[:, :, 0] >= 0) & (long_coords[:, :, 0] < self.nx[0]) & \
                (long_coords[:, :, 1] >= 0) & (long_coords[:, :, 1] < self.nx[1]) & \
                (long_coords[:, :, 2] >= 0) & (long_coords[:, :, 2] < self.nx[2])
        
        bev_map = torch.zeros((B, self.data_conf['bev_grid_shape'][0], self.data_conf['bev_grid_shape'][1], feats.shape[-1]), device=feats.device)
        
        for b in range(B):
            cur_valid = valid[b]
            cur_coords = long_coords[b][cur_valid]
            cur_feats = feats[b][cur_valid]
            
            if cur_coords.shape[0] == 0:
                continue
            
            bev_map[b].index_put_((cur_coords[:, 0], cur_coords[:, 1]), cur_feats, accumulate=True)

        return bev_map.permute(0, 3, 1, 2).contiguous()

    def forward(self, x, rots, trans, intrinsics):
        B, N, C, D, H, W = x.shape
        x = x.permute(0, 1, 3, 4, 5, 2) # (B, N, D, H, W, C)
        geom = self.get_geometry(rots, trans, intrinsics) 
        bev = self.voxel_pooling(geom, x)
        return bev


class BEVModel(nn.Module):
    def __init__(self, conf):
        super(BEVModel, self).__init__()
        self.conf = conf
        self.C = 64 # Каналы признаков
        self.D = int((conf['d_bound'][1] - conf['d_bound'][0]) / conf['d_bound'][2])
        
        self.cam_encoder = CamEncoder(self.D, self.C, downsample=32)
        self.view_transformer = ViewTransformer(conf, conf)
        
        # Декодер BEV (простой U-Net like или ResNet блоки)
        # Вход: 64 канала. Выход: 1 канал (logits)
        self.bev_decoder = nn.Sequential(
            nn.Conv2d(self.C, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1) # Итоговый предиктор
        )

    def forward(self, images, intrinsics, cam2cars):
        # images: (B, 4, 3, H, W)
        B, N, C, H, W = images.shape
        
        # 1. CamEncoder
        # Сливаем батч и камеры: (B*N, 3, H, W)
        images_flat = images.view(B*N, C, H, W)
        depth_dist, context = self.cam_encoder(images_flat)
        
        # Outer product: (B*N, C, D, H_f, W_f)
        # context: (B*N, C, 8, 16), depth: (B*N, D, 8, 16)
        x = context.unsqueeze(2) * depth_dist.unsqueeze(1)
        
        # Reshape обратно в батч
        # (B, N, C, D, H_f, W_f)
        x = x.view(B, N, self.C, self.D, x.shape[-2], x.shape[-1])
        
        # 2. View Transformer (Lift & Splat)
        # Разбираем cam2cars на rotation и translation
        rots = cam2cars[:, :, :3, :3]
        trans = cam2cars[:, :, :3, 3]
        
        bev_feat = self.view_transformer(x, rots, trans, intrinsics)
        
        # 3. BEV Decoder (Shoot)
        logits = self.bev_decoder(bev_feat) # (B, 1, 188, 126)
        
        return logits
