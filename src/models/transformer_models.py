import math

import torch
import torch.nn as nn
from torchvision.models import resnet50, swin_t, Swin_T_Weights

from .utils import positionalencoding2d


class MultiViewBEVModelResNet(nn.Module):
    def __init__(self, bev_h=48, bev_w=32, num_classes=1, num_cameras=4):
        super().__init__()
        
        # --- Backbone ---
        resnet = resnet50(weights='DEFAULT')
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        self.neck = nn.Conv2d(2048, 512, kernel_size=1)

        # --- BEV queries ---
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_classes = num_classes
        self.num_cameras = num_cameras
        
        self.bev_queries = nn.Parameter(torch.empty(bev_h * bev_w, 512))
        nn.init.xavier_normal_(self.bev_queries)
        
        # --- Transformer ---
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(512, 8, batch_first=True),
            num_layers=4,
        )
        
        self.register_buffer("pos_embed", positionalencoding2d(8, 16, 512))  # H_feat=8, W_feat=16
        self.register_buffer("bev_pos_embed", positionalencoding2d(self.bev_h, self.bev_w, 512) )

        self.upsample = nn.Sequential(
            # Stage 1: coarse → medium (~48x32)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(96,64), mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            # Stage 2: medium → finer (~94x63)
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(188,126), mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            # Stage 3: finer → final (188x126)
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # Final projection
            nn.BatchNorm2d(64),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

        self.intrinsics_encoder = nn.Sequential(
            nn.BatchNorm1d(12),
            nn.Linear(12, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.car2cams_encoder = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.Linear(16, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )


    def forward(self, x, intrinsics, car2cams):
        B, N, C, H, W = x.shape
        assert N == self.num_cameras, "Number of cameras mismatch"
        
        intrinsics = self.intrinsics_encoder(intrinsics.reshape(B * N, -1).float()).reshape(B, N, -1)
        car2cams = self.car2cams_encoder(car2cams.reshape(B * N, -1).float()).reshape(B, N, -1)

        # --- Extract features from each camera ---
        x = x.view(B * N, C, H, W)
        features = self.encoder(x)  # (B*N, 512, 8,16)
        features = self.neck(features)
        _, C_feat, H_feat, W_feat = features.shape
        features = features.view(B, N, C_feat, H_feat, W_feat)
        
        # --- Flatten camera + spatial dims for transformer ---
        features = features.permute(0, 1, 3, 4, 2)  # (B, N, H_feat, W_feat, C)
        features = features + self.pos_embed[None, None]  # (B, N, H_feat, W_feat, C)
        features = features.reshape(B, N, H_feat*W_feat, C_feat)  # (B, N, H_feat*W_feat, 512)
        features = features + intrinsics.unsqueeze(2) + car2cams.unsqueeze(2) 
        features = features.reshape(B, N*H_feat*W_feat, C_feat)  # (B, seq_len, 512)

        bev_tokens = self.bev_queries.reshape(self.bev_h, self.bev_w, C_feat) + self.bev_pos_embed
        bev_tokens = bev_tokens.view(self.bev_h*self.bev_w, C_feat)
        # --- Transformer decoder ---
        bev_tokens = self.transformer(
            self.bev_queries.unsqueeze(0).repeat(B,1,1),  # (B, bev_h*bev_w, 512)
            features
        )  # (B, bev_h*bev_w, 512)
        
        # --- Reshape to coarse BEV map ---
        bev_tokens = bev_tokens.transpose(1, 2).view(B, 512, self.bev_h, self.bev_w)  # (B, 512, H_bev, W_bev)
        
        # --- Conv upsampling to final BEV ---
        bev_logits = self.upsample(bev_tokens)  # (B, num_classes, 188, 126)
        
        return bev_logits


class MultiViewBEVModelSwin(nn.Module):
    def __init__(self, bev_h=48, bev_w=32, num_classes=1, num_cameras=4):
        super().__init__()
        
        # --- Backbone ---
        swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(swin.children())[:-3])
        
        self.neck = nn.Conv2d(768, 512, kernel_size=1)

        # --- BEV queries ---
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_classes = num_classes
        self.num_cameras = num_cameras
        
        self.bev_queries = nn.Parameter(torch.empty(bev_h * bev_w, 512))
        nn.init.xavier_normal_(self.bev_queries)
        
        # --- Transformer ---
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(512, 8, batch_first=True),
            num_layers=4,
        )
        
        self.register_buffer("pos_embed", positionalencoding2d(8, 16, 512))  # H_feat=8, W_feat=16
        self.register_buffer("bev_pos_embed", positionalencoding2d(self.bev_h, self.bev_w, 512) )

        self.upsample = nn.Sequential(
            # Stage 1: coarse → medium (~48x32)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(96,64), mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            # Stage 2: medium → finer (~94x63)
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(188,126), mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            # Stage 3: finer → final (188x126)
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # Final projection
            nn.BatchNorm2d(64),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

        self.intrinsics_encoder = nn.Sequential(
            nn.BatchNorm1d(12),
            nn.Linear(12, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.car2cams_encoder = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.Linear(16, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )


    def forward(self, x, intrinsics, car2cams):
        B, N, C, H, W = x.shape
        assert N == self.num_cameras, "Number of cameras mismatch"
        
        intrinsics = self.intrinsics_encoder(intrinsics.reshape(B * N, -1).float()).reshape(B, N, -1)
        car2cams = self.car2cams_encoder(car2cams.reshape(B * N, -1).float()).reshape(B, N, -1)

        # --- Extract features from each camera ---
        x = x.view(B * N, C, H, W)
        features = self.encoder(x)  # (B*N, 512, 8,16)
        features = self.neck(features)
        _, C_feat, H_feat, W_feat = features.shape
        features = features.view(B, N, C_feat, H_feat, W_feat)
        
        # --- Flatten camera + spatial dims for transformer ---
        features = features.permute(0, 1, 3, 4, 2)  # (B, N, H_feat, W_feat, C)
        features = features + self.pos_embed[None, None]  # (B, N, H_feat, W_feat, C)
        features = features.reshape(B, N, H_feat*W_feat, C_feat)  # (B, N, H_feat*W_feat, 512)
        features = features + intrinsics.unsqueeze(2) + car2cams.unsqueeze(2) 
        features = features.reshape(B, N*H_feat*W_feat, C_feat)  # (B, seq_len, 512)

        bev_tokens = self.bev_queries.reshape(self.bev_h, self.bev_w, C_feat) + self.bev_pos_embed
        bev_tokens = bev_tokens.view(self.bev_h*self.bev_w, C_feat)
        # --- Transformer decoder ---
        bev_tokens = self.transformer(
            self.bev_queries.unsqueeze(0).repeat(B,1,1),  # (B, bev_h*bev_w, 512)
            features
        )  # (B, bev_h*bev_w, 512)
        
        # --- Reshape to coarse BEV map ---
        bev_tokens = bev_tokens.transpose(1, 2).view(B, 512, self.bev_h, self.bev_w)  # (B, 512, H_bev, W_bev)
        
        # --- Conv upsampling to final BEV ---
        bev_logits = self.upsample(bev_tokens)  # (B, num_classes, 188, 126)
        
        return bev_logits
