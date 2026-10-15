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

class MultiScaleFusion(nn.Module):
    """多尺度特征金字塔融合模块 - 适配160通道版本"""
    def __init__(self, img_channels=160, pts_channels=256, out_channels=160):
        super(MultiScaleFusion, self).__init__()
        
        # 图像BEV下采样金字塔 (输入256x256)
        self.img_pyramid = nn.ModuleList([
            nn.Identity(),  # 原始尺度: 256x256
            nn.Sequential(  # 1/2尺度: 128x128
                nn.Conv2d(img_channels, img_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(img_channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(  # 1/4尺度: 64x64
                nn.Conv2d(img_channels, img_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(img_channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(  # 1/8尺度: 32x32
                nn.Conv2d(img_channels, img_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(img_channels),
                nn.ReLU(inplace=True)
            )
        ])
        
        # 点云特征通道调整（将256通道调整为160通道）
        self.pts_channel_adapters = nn.ModuleList([
            nn.Sequential(  # 44x44 -> 32x32 尺度
                nn.Conv2d(pts_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)
            ),
            nn.Sequential(  # 44x44 -> 64x64 尺度
                nn.Conv2d(pts_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)
            ),
            nn.Sequential(  # 44x44 -> 128x128 尺度
                nn.Conv2d(pts_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
            ),
            nn.Sequential(  # 44x44 -> 256x256 尺度
                nn.Conv2d(pts_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
            )
        ])
        
        # 逐尺度融合层
        self.scale_fusions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])
        
        # 最终融合层（输入4个160通道的特征，输出160通道）
        self.final_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 残差连接增强原始图像特征
        self.residual_conv = nn.Sequential(
            nn.Conv2d(img_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, img_feat, pts_feat):
        """
        Args:
            img_feat: [B, 160, 256, 256] 图像BEV特征
            pts_feat: [B, 256, 44, 44] 点云BEV特征
        
        Returns:
            fused_feat: [B, 160, 256, 256] 融合后的特征
        """
        # 构建图像特征金字塔
        img_features = [img_feat]  # [B, 160, 256, 256]
        for downsample in self.img_pyramid[1:]:
            img_features.append(downsample(img_features[-1]))
        # img_features现在包含: [256x256, 128x128, 64x64, 32x32] 四个尺度
        
        # 处理点云特征并构建金字塔（调整通道数和上采样到对应尺度）
        pts_features = []
        for i, adapter in enumerate(self.pts_channel_adapters):
            pts_feat_adapted = adapter(pts_feat)
            pts_features.append(pts_feat_adapted)
        # pts_features现在包含: [32x32, 64x64, 128x128, 256x256] 四个尺度（都是160通道）
        
        # 逐尺度融合
        fused_features = []
        for i, (img_feat_scale, pts_feat_scale) in enumerate(zip(img_features, pts_features)):
            # 确保尺寸匹配
            if img_feat_scale.shape[-2:] != pts_feat_scale.shape[-2:]:
                pts_feat_scale = F.interpolate(
                    pts_feat_scale, 
                    size=img_feat_scale.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # 通道数都是160，可以相加融合
            fused_scale = img_feat_scale + pts_feat_scale
            
            # 进一步融合
            fused_scale = self.scale_fusions[i](fused_scale)
            
            fused_features.append(fused_scale)
        
        # 将所有融合后的特征上采样到最大尺度(256x256)并拼接
        target_size = (256, 256)
        upsampled_features = []
        for feat in fused_features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            upsampled_features.append(feat)
        
        # 最终融合
        final_feat = torch.cat(upsampled_features, dim=1)  # [B, 640, 256, 256]
        final_feat = self.final_fusion(final_feat)  # [B, 160, 256, 256]
        
        # 添加残差连接，保留原始图像特征的信息
        img_feat_residual = self.residual_conv(img_feat)  # [B, 160, 256, 256]
        final_feat = final_feat + img_feat_residual  # 残差连接
        
        return final_feat


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
        
        # 使用多尺度融合模块替代原来的融合层
        self.fusion_layer = MultiScaleFusion(
            img_channels=160,  # 修改为160通道
            pts_channels=256, 
            out_channels=160   # 输出保持160通道
        )
        
        self.is_train_height = is_train_height

    def forward(
        self,
        x,
        points,
        mats_dict,
        timestamps=None,
    ):
        """Forward function for BEVHeightPlus

        Args:
            x (Tensor): Input ferature map.
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps (long): Timestamp.
                Default: None.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        if self.is_train_height and self.training:
            x_img = self.backbone(x,
                              mats_dict,
                              timestamps,
                              is_return_height=True)
            
            # 提取点云特征
            x_pts = self.extract_pts_feat(points)  # [B, 256, 44, 44]
            
            # 多尺度融合
            x_fusion = self.fusion_layer(x_img[0], x_pts)  # [B, 160, 256, 256]
            
            preds = self.head(x_fusion)
            return preds, x_img[1:]
        else:
            # 提取图像特征
            x_img = self.backbone(x, mats_dict, timestamps) # torch.Size([B, 160, 256, 256])
            
            # 提取点云特征
            x_pts = self.extract_pts_feat(points)  # [B, 256, 44, 44]
            
            # 多尺度融合
            x_fusion = self.fusion_layer(x_img, x_pts)  # [B, 160, 256, 256]
            
            preds = self.head(x_fusion)
            return preds
        
    def extract_pts_feat(self, points):
        points = [point.float() for point in points]
        feats, coords, sizes = self.voxelize(points) # 体素化
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
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        return self.head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        """Loss function for BEVHeightPlus.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.head.loss(targets, preds_dicts)

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)