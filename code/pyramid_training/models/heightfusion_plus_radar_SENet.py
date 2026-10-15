'''
-*- coding: utf-8 -*-
Author: Bowen Wang <98.wangbowen@sjtu.edu.cn>; Zhenhao Wang <wangzh2024@sjtu.edu.cn>
License: TDG-Attribution-NonCommercial-NoDistrib

intermediate heter fusion dataset

Each agent should retrieve the objects itself, and merge them by iou, 
instead of using the cooperative label.
''' 

from torch import nn
import torch
from torch.nn import functional as F
from layers.backbones.lss_fpn import LSSFPN
from layers.heads.heightfusion_head import HeightfusionHead
from layers.sparse_encoder import BEVFusionSparseEncoder
from .ops import Voxelization
__all__ = ['BEVHeightPlus']

class SEBlock(nn.Module):
    """改进的Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 同时考虑最大池化
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的权重参数

    def forward(self, x):
        b, c, _, _ = x.size()
        # 使用平均池化和最大池化
        avg_y = self.avg_pool(x).view(b, c)
        max_y = self.max_pool(x).view(b, c)
        y = torch.cat([avg_y, max_y], dim=1)
        y = self.fc(y).view(b, c, 1, 1)
        # 使用可学习权重控制注意力强度
        return x * (1 + self.gamma * y)


class AttentionFusionBlock(nn.Module):
    """注意力融合模块"""
    def __init__(self, channels):
        super(AttentionFusionBlock, self).__init__()
        self.channels = channels
        
        # 模态权重学习
        self.modal_attention = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 1),
            nn.Softmax(dim=1)  # 输出三个模态的权重
        )
        
        # 通道注意力
        self.channel_attention = SEBlock(channels)

    def forward(self, img_feat, pts_feat, radar_feat):
        # 确保特征尺寸一致
        if pts_feat.shape[-2:] != img_feat.shape[-2:]:
            pts_feat = F.interpolate(pts_feat, size=img_feat.shape[-2:], 
                                   mode='bilinear', align_corners=False)
        if radar_feat.shape[-2:] != img_feat.shape[-2:]:
            radar_feat = F.interpolate(radar_feat, size=img_feat.shape[-2:], 
                                     mode='bilinear', align_corners=False)
        
        # 计算模态权重
        concat_features = torch.cat([img_feat, pts_feat, radar_feat], dim=1)
        modal_weights = self.modal_attention(concat_features)  # [B, 3, H, W]
        
        # 加权融合
        modal_weights = modal_weights.unsqueeze(2)  # [B, 3, 1, H, W]
        features = torch.stack([img_feat, pts_feat, radar_feat], dim=1)  # [B, 3, C, H, W]
        weighted_features = features * modal_weights
        fused_feat = weighted_features.sum(dim=1)  # [B, C, H, W]
        
        # 通道注意力
        fused_feat = self.channel_attention(fused_feat)
        
        return fused_feat


class ProgressiveSENetFusion(nn.Module):
    """渐进式SENet融合 - 保留多尺度特性"""
    def __init__(self, img_channels=160, pts_channels=256, out_channels=160):
        super(ProgressiveSENetFusion, self).__init__()
        
        # 图像特征金字塔（与原始保持一致）
        self.img_pyramid = nn.ModuleList([
            nn.Identity(),  # 原始尺度: 256x256
            nn.Sequential(
                nn.Conv2d(img_channels, img_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(img_channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(img_channels, img_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(img_channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(img_channels, img_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(img_channels),
                nn.ReLU(inplace=True)
            )
        ])
        
        # 点云特征金字塔 - 使用渐进式上采样
        self.pts_pyramid = nn.ModuleList([
            self.build_pts_branch(pts_channels, out_channels, scale=32),  # 44->32
            self.build_pts_branch(pts_channels, out_channels, scale=64),  # 44->64
            self.build_pts_branch(pts_channels, out_channels, scale=128), # 44->128
            self.build_pts_branch(pts_channels, out_channels, scale=256)  # 44->256
        ])
        
        # 毫米波雷达金字塔（同点云）
        self.radar_pyramid = nn.ModuleList([
            self.build_pts_branch(pts_channels, out_channels, scale=32),
            self.build_pts_branch(pts_channels, out_channels, scale=64),
            self.build_pts_branch(pts_channels, out_channels, scale=128),
            self.build_pts_branch(pts_channels, out_channels, scale=256)
        ])
        
        # 逐尺度注意力融合模块
        self.scale_fusions = nn.ModuleList([
            AttentionFusionBlock(out_channels) for _ in range(4)
        ])
        
        # 多尺度SE注意力
        self.multi_scale_se = nn.ModuleList([
            SEBlock(out_channels, reduction=8) for _ in range(4)
        ])
        
        # 渐进式特征融合
        self.progressive_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                SEBlock(out_channels, reduction=8)
            ) for _ in range(3)
        ])
        
        # 最终输出
        self.final_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels, reduction=8)
        )
        
        # 上采样层
        self.upsample_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            for _ in range(3)
        ])

    def build_pts_branch(self, in_channels, out_channels, scale):
        """构建点云特征处理分支"""
        layers = []
        # 先调整通道数
        layers.append(nn.Conv2d(in_channels, out_channels, 1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # 计算需要的上采样倍数
        target_scale = scale
        current_scale = 44
        
        # 渐进式上采样
        while current_scale < target_scale:
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            current_scale *= 2
            
        # 最后微调到目标尺寸
        if current_scale != target_scale:
            layers.append(nn.Upsample(size=(target_scale, target_scale), 
                                    mode='bilinear', align_corners=False))
        
        return nn.Sequential(*layers)

    def forward(self, img_feat, pts_feat, radar_feat):
        """
        Args:
            img_feat: [B, 160, 256, 256] 图像BEV特征
            pts_feat: [B, 256, 44, 44] 激光雷达点云BEV特征
            radar_feat: [B, 256, 44, 44] 毫米波点云BEV特征
        """
        # 1. 构建图像特征金字塔
        img_pyramid = [img_feat]
        for downsample in self.img_pyramid[1:]:
            img_pyramid.append(downsample(img_pyramid[-1]))
        
        # 2. 处理点云和雷达特征
        pts_pyramid = []
        radar_pyramid = []
        for i in range(4):
            pts_pyramid.append(self.pts_pyramid[i](pts_feat))
            radar_pyramid.append(self.radar_pyramid[i](radar_feat))
        
        # 3. 逐尺度融合
        scale_outputs = []
        for i in range(4):
            # 注意力融合
            fused = self.scale_fusions[i](img_pyramid[i], pts_pyramid[i], radar_pyramid[i])
            # 尺度SE注意力
            fused = self.multi_scale_se[i](fused)
            scale_outputs.append(fused)
        
        # 4. 渐进式融合（从最小尺度开始）
        progressive_features = []
        current_feat = scale_outputs[3]  # 从最小尺度开始（32x32）
        progressive_features.append(current_feat)
        
        # 向上融合
        for i in range(3):
            # 上采样
            upsampled = self.upsample_layers[i](current_feat)
            # 与下一尺度特征融合
            next_feat = scale_outputs[2 - i]
            combined = torch.cat([upsampled, next_feat], dim=1)
            current_feat = self.progressive_fusion[i](combined)
            progressive_features.append(current_feat)
        
        # 现在progressive_features包含所有尺度的特征，从大到小
        # 反转顺序，使大尺度在前
        progressive_features = progressive_features[::-1]
        
        # 5. 最终多尺度融合
        # 将所有特征上采样到最大尺度
        target_size = img_feat.shape[-2:]  # 256x256
        aligned_features = []
        for feat in progressive_features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)
        
        # 拼接所有尺度的特征
        final_concat = torch.cat(aligned_features, dim=1)  # [B, 640, 256, 256]
        final_output = self.final_fusion(final_concat)  # [B, 160, 256, 256]
        
        # 6. 残差连接（保留原始图像特征）
        final_output = final_output + img_feat
        
        return final_output


class BEVHeightPlus(nn.Module):
    """
    Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
        is_train_height (bool): Whether to return height.
            Default: False.
    """

    # TODO: Reduce grid_conf and data_aug_conf
    def __init__(self, backbone_conf, head_conf, pts_conf, voxelize_cfg, is_train_height=False):
        super(BEVHeightPlus, self).__init__()
        self.backbone = LSSFPN(**backbone_conf)
        self.head = HeightfusionHead(**head_conf)
        
        self.pts_middle_encoder = BEVFusionSparseEncoder(**pts_conf)
        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)
        
        # 使用渐进式SENet融合
        self.fusion_layer = ProgressiveSENetFusion(
            img_channels=160,
            pts_channels=256,
            out_channels=160
        )
        
        self.is_train_height = is_train_height

    def forward(
        self,
        x,
        points,
        points_radar,
        mats_dict,
        timestamps=None,
    ):
        """Forward function for BEVHeightPlus"""
        if self.is_train_height and self.training:
            x_img = self.backbone(x, mats_dict, timestamps, is_return_height=True)
            
            # 提取点云特征
            x_pts = self.extract_pts_feat(points)  # [B, 256, 44, 44]
            
            # 提取毫米波点云特征
            x_pts_radar = self.extract_pts_feat(points_radar)
            
            # 渐进式SENet融合
            x_fusion = self.fusion_layer(x_img[0], x_pts, x_pts_radar)
            
            preds = self.head(x_fusion)
            return preds, x_img[1:]
        else:
            # 提取图像特征
            x_img = self.backbone(x, mats_dict, timestamps)
            
            # 提取点云特征
            x_pts = self.extract_pts_feat(points)
            
            # 提取毫米波点云特征
            x_pts_radar = self.extract_pts_feat(points_radar)
            
            # 渐进式SENet融合
            x_fusion = self.fusion_layer(x_img, x_pts, x_pts_radar)
            
            preds = self.head(x_fusion)
            return preds
        
    def extract_pts_feat(self, points):
        points = [point.float() for point in points]
        feats, coords, sizes = self.voxelize(points)
        batch_size = coords[-1, 0] + 1
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x
    
    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def get_targets(self, gt_boxes, gt_labels):
        """Generate training targets for a single sample."""
        return self.head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        """Loss function for BEVHeightPlus."""
        return self.head.loss(targets, preds_dicts)

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions."""
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)