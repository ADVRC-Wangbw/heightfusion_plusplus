'''
-*- coding: utf-8 -*-
Author: Bowen Wang <98.wangbowen@sjtu.edu.cn>; Zhenhao Wang <wangzh2024@sjtu.edu.cn>
License: TDG-Attribution-NonCommercial-NoDistrib

intermediate heter fusion dataset

Each agent should retrieve the objects itself, and merge them by iou, 
instead of using the cooperative label.
''' 

from argparse import ArgumentParser, Namespace

import os
import mmcv
import pytorch_lightning as pl
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import torchvision.models as models
from pytorch_lightning.core import LightningModule
from torch.cuda.amp.autocast_mode import autocast
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import re

from dataset.nusc_mv_det_dataset_radar_SENet import NuscMVDetDataset, collate_fn
from evaluators.det_evaluators import RoadSideEvaluator
from models.heightfusion_plus_radar_SENet import BEVHeightPlus
from utils.torch_dist import all_gather_object, get_rank, synchronize
from utils.backup_files import backup_codebase

H = 384
W = 1280
final_dim = (384, 1280)
img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)
model_type = 2 # 0: BEVDepth, 1: BEVHeight, 2: BEVHeight++

return_depth = True # 是否在训练时返回深度/高度监督

data_root = None
bev_dim = 160 if model_type==2 else 80
 
backbone_conf = {
    'x_bound': [0, 102.4, 0.4],
    'y_bound': [-51.2, 51.2, 0.4],
    'z_bound': [-5, 3, 8],
    'd_bound': [1.0, 102.0, 0.5],
    'h_bound': [0.0, 4.0, 80],
    'model_type': model_type,
    'final_dim':
    final_dim,
    'output_channels':
    80,
    'downsample_factor':
    16,
    'img_backbone_conf':
    dict(
        type='ResNet',
        depth=101,
        frozen_stages=0,
        out_indices=[0, 1, 2, 3],
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'),
    ),
    'img_neck_conf':
    dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128],
    ),
    'height_net_conf':
    dict(in_channels=512, mid_channels=512)
}
ida_aug_conf = {
    'final_dim':
    final_dim,
    'H':
    H,
    'W':
    W,
    'bot_pct_lim': (0.0, 0.0),
    'cams': ['CAM_FRONT'],
    'Ncams': 1,
}

bev_backbone = dict(
    type='ResNet',
    in_channels = bev_dim,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels= bev_dim * 2,
)

bev_neck = dict(type='SECONDFPN',
                in_channels=[bev_dim, bev_dim * 2, bev_dim * 4, bev_dim * 8],
                upsample_strides=[1, 2, 4, 8],
                out_channels=[64, 64, 64, 64])

CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
    'bicycle'
]

TASKS = [
    dict(num_class=1, class_names=['car']),
    dict(num_class=2, class_names=['truck', 'construction_vehicle']),
    dict(num_class=2, class_names=['bus', 'trailer']),
    dict(num_class=1, class_names=['barrier']),
    dict(num_class=2, class_names=['motorcycle', 'bicycle']),
    dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
    dict(num_class=1, class_names=['bicycle']),
]

common_heads = dict(reg=(2, 2),
                    height=(1, 2),
                    dim=(3, 2),
                    rot=(2, 2),
                    vel=(2, 2))

bbox_coder = dict(
    type='CenterPointBBoxCoder',
    post_center_range=[0.0, -51.2, -10.0, 102.4, 51.2, 10.0],
    max_num=500,
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.1, 0.1, 8],
    pc_range=[0, -51.2, -5, 102.4, 51.2, 3],
    code_size=9,
)

train_cfg = dict(
    point_cloud_range=[0, -51.2, -5, 102.4, 51.2, 3],
    grid_size=[1024, 1024, 1],
    voxel_size=[0.1, 0.1, 8],
    out_size_factor=4,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
)

test_cfg = dict(
    post_center_limit_range=[0.0, -51.2, -10.0, 102.4, 51.2, 10.0],
    max_per_img=500,
    max_pool_nms=False,
    min_radius=[4, 12, 10, 1, 0.85, 0.175, 0.85],
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.1, 0.1, 8],
    nms_type='circle',
    pre_max_size=1000,
    post_max_size=83,
    nms_thr=0.2,
)

head_conf = {
    'bev_backbone_conf': bev_backbone,
    'bev_neck_conf': bev_neck,
    'tasks': TASKS,
    'common_heads': common_heads,
    'bbox_coder': bbox_coder,
    'train_cfg': train_cfg,
    'test_cfg': test_cfg,
    'in_channels': 256,  # Equal to bev_neck output_channels.
    'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
    'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
    'gaussian_overlap': 0.1,
    'min_radius': 2,
}

point_cloud_range=[0, -70.4, -5, 140.8, 70.4, 3]
voxel_size = [0.4, 0.4, 0.2]
sparse_shape = [352, 352, 41]
pts_conf = dict(
        in_channels=4,
        sparse_shape=sparse_shape,
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
        block_type='basicblock')
voxelize_cfg=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=[120000, 160000],
            voxelize_reduce=True)

class HeightfusionplusLightningModel(LightningModule):
    MODEL_NAMES = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith('__')
                         and callable(models.__dict__[name]))

    def __init__(self,
                 gpus: int = 1,
                 data_root=data_root,
                 eval_interval=1,
                 batch_size_per_device=2,
                 class_names=CLASSES,
                 backbone_conf=backbone_conf,
                 head_conf=head_conf,
                 ida_aug_conf=ida_aug_conf,
                 pts_conf=pts_conf,
                 voxelize_cfg=voxelize_cfg,
                 default_root_dir='outputs/heightfusion++_radar',
                 pretrained_path=None,
                 lidar_pretrained_path=None,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.gpus = gpus
        self.eval_interval = eval_interval
        self.batch_size_per_device = batch_size_per_device
        self.data_root = data_root
        self.learning_rate = 3e-5
        self.class_names = class_names
        self.backbone_conf = backbone_conf
        self.head_conf = head_conf
        self.ida_aug_conf = ida_aug_conf
        self.pts_conf = pts_conf
        self.voxelize_cfg = voxelize_cfg
        self.return_depth = return_depth
        mmcv.mkdir_or_exist(default_root_dir)
        self.default_root_dir = default_root_dir
        self.evaluator = RoadSideEvaluator(class_names=self.class_names,
                                           current_classes=["Car", "Pedestrian", "Cyclist"],
                                           data_root=data_root,
                                           gt_label_path=os.path.join(self.data_root, "training/label_2"),
                                           output_dir=self.default_root_dir)
        self.model = BEVHeightPlus(self.backbone_conf, self.head_conf, self.pts_conf, self.voxelize_cfg, is_train_height=self.return_depth)
        self.mode = 'valid'
        self.img_conf = img_conf
        self.data_use_cbgs = False
        self.num_sweeps = 1
        self.sweep_idxes = list()
        self.key_idxes = list()
        self.downsample_factor = self.backbone_conf['downsample_factor']
        self.dbound = self.backbone_conf['d_bound']
        self.hbound = self.backbone_conf['h_bound']
        self.height_channels = int(self.hbound[2])
        self.depth_channels = int((self.dbound[1] - self.dbound[0]) / self.dbound[2])
        self.val_list = [x.strip() for x in open(os.path.join(data_root, "ImageSets",  "val.txt")).readlines()]
        
        self.pretrained_path = pretrained_path
        self.lidar_pretrained_path = lidar_pretrained_path
        if self.pretrained_path and os.path.exists(self.pretrained_path):
            self.load_pretrained_weights(self.pretrained_path)
        if self.lidar_pretrained_path and os.path.exists(self.lidar_pretrained_path):
            self.load_lidar_pretrained_weights(self.lidar_pretrained_path)
      
    def load_pretrained_weights(self, pretrained_path):
        """加载预训练权重，只加载匹配的部分"""
        print(f"Loading pretrained weights from {pretrained_path}")
        
        # 加载预训练权重的state_dict
        checkpoint = torch.load(pretrained_path, map_location='cuda')
        
        # 处理不同格式的checkpoint
        if 'state_dict' in checkpoint:
            pretrained_state_dict = checkpoint['state_dict']
            
        else:
            pretrained_state_dict = checkpoint
        
        # 获取当前模型的state_dict
        current_state_dict = self.model.state_dict()
        
        # 筛选可以加载的权重
        loaded_keys = []
        skipped_keys = []
        
        for key, param in pretrained_state_dict.items():
            # 处理可能的key前缀（如'model.'）
            if key.startswith('model.'):
                clean_key = key[6:]  # 去掉'model.'前缀
            else:
                clean_key = key
                
            # 检查key是否存在且形状匹配
            if clean_key in current_state_dict and param.shape == current_state_dict[clean_key].shape:
                current_state_dict[clean_key] = param
                loaded_keys.append(clean_key)
            else:
                skipped_keys.append(key)
        
        # 加载匹配的权重
        self.model.load_state_dict(current_state_dict, strict=False) # strict=False 允许部分加载
        
        print(f"Successfully loaded {len(loaded_keys)} parameters")
        print(f"Skipped {len(skipped_keys)} parameters (shape mismatch or not found)")
        
        
    def load_lidar_pretrained_weights(self, lidar_pretrained_path):
        """加载雷达预训练权重"""
        print(f"Loading lidar pretrained weights from {lidar_pretrained_path}")
        
        # 加载雷达checkpoint
        lidar_checkpoint = torch.load(lidar_pretrained_path, map_location='cuda')
        
        # 处理不同格式的checkpoint
        if 'state_dict' in lidar_checkpoint:
            lidar_state_dict = lidar_checkpoint['state_dict']
        else:
            lidar_state_dict = lidar_checkpoint
        
        # 获取当前模型的state_dict
        current_state_dict = self.model.state_dict()
        
        # 筛选可以加载的权重
        loaded_keys = []
        skipped_keys = []
        
        for key, param in lidar_state_dict.items():
            # 处理可能的key前缀（如'model.'）
            if key[:23] == 'encoders.lidar.backbone':
                a = key[23:]
                key = 'model.pts_middle_encoder' + a
            if key.startswith('model.'):
                clean_key = key[6:]  # 去掉'model.'前缀
            else:
                clean_key = key
                
            # model.pts_middle_encoder == encoders.lidar.backbone
            
            # 检查key是否存在且形状匹配
            if clean_key in current_state_dict and param.shape == current_state_dict[clean_key].shape:
                current_state_dict[clean_key] = param
                loaded_keys.append(clean_key)
            else:
                skipped_keys.append(key)
        
        # 加载匹配的权重
        self.model.load_state_dict(current_state_dict, strict=False) # strict=False 允许部分加载
        
        print(f"Successfully loaded {len(loaded_keys)} parameters")
        print(f"Skipped {len(skipped_keys)} parameters (shape mismatch or not found)")



    def is_inval(self, img_metas):
        for img_meta in img_metas:
            if img_meta['token'].split("/")[1] in self.val_list:
                return True            
        return False

    def forward(self, sweep_imgs, points, points_radar, mats):
        return self.model(sweep_imgs, points, points_radar, mats)

    def training_step(self, batch):
        if self.return_depth:
            (sweep_imgs, points, points_radar, mats, timestamps, img_metas, gt_boxes, gt_labels, depth_labels, height_labels) = batch
        else:
            (sweep_imgs, points, points_radar, mats, timestamps, img_metas, gt_boxes, gt_labels) = batch
        
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            
            sweep_imgs = sweep_imgs.cuda()
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]

        if self.return_depth:
            if model_type == 0:
                preds, depth_preds = self(sweep_imgs, points, points_radar, mats)
                depth_preds = depth_preds[0]
            elif model_type == 1:
                preds, height_preds = self(sweep_imgs, points, points_radar, mats)
                height_preds = height_preds[0]
            elif model_type == 2:
                preds, geometry_preds = self(sweep_imgs, points, points_radar, mats)
                depth_preds, height_preds = geometry_preds[0], geometry_preds[1]
        else:
            preds = self(sweep_imgs, points, points_radar, mats)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            targets = self.model.module.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.module.loss(targets, preds)
        else:
            targets = self.model.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.loss(targets, preds)
        self.log('detection_loss', detection_loss)
        if self.return_depth:
            if len(depth_labels.shape) == 5 and model_type in [0, 2]:
                depth_labels = depth_labels[:, 0, ...]
            if len(height_labels.shape) == 5 and model_type in [1, 2]:
                height_labels = height_labels[:, 0, ...]
            if model_type == 0:
                depth_loss = self.get_depth_loss(depth_labels.cuda(), depth_preds)
                self.log('depth_loss', depth_loss)
                return detection_loss + depth_loss
            elif model_type == 1:
                height_loss = self.get_height_loss(height_labels.cuda(), height_preds)
                self.log('height_loss', height_loss)
                return detection_loss + height_loss
            elif model_type == 2:
                depth_loss = self.get_depth_loss(depth_labels.cuda(), depth_preds)
                height_loss = self.get_height_loss(height_labels.cuda(), height_preds)
                self.log('depth_loss', depth_loss)
                self.log('height_loss', height_loss)
                if self.is_inval(img_metas):
                    return depth_loss + height_loss
                else:
                    return detection_loss + depth_loss + height_loss
        else:
            return detection_loss

    def training_epoch_end(self, outputs) -> None:
        """在每个训练轮结束时，计算本轮平均loss并追加到文件。

        `outputs` 可能是 loss 张量列表或包含 loss 的 dict 列表。
        仅在主进程（rank 0）写文件以避免多进程重复写入。
        """
        # 收集每个step的loss数值
        losses = []
        for out in outputs:
            try:
                # Lightning 有时返回张量，有时返回 dict
                if isinstance(out, dict) and 'loss' in out:
                    val = out['loss']
                else:
                    val = out
                if hasattr(val, 'detach'):
                    val = val.detach().cpu().item()
                else:
                    val = float(val)
                losses.append(val)
            except Exception:
                continue

        if len(losses) == 0:
            return

        avg_loss = sum(losses) / len(losses)

        # 只在主进程写入文件
        try:
            if get_rank() == 0:
                os.makedirs(self.default_root_dir, exist_ok=True)
                log_path = os.path.join(self.default_root_dir, 'epoch_loss.txt')
                with open(log_path, 'a') as f:
                    f.write(f"{self.current_epoch}\t{avg_loss:.6f}\n")
        except Exception:
            pass


    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return 3.0 * depth_loss

    def get_height_loss(self, height_labels, height_preds):
        height_labels = self.get_downsampled_gt_height(height_labels)
        height_preds = height_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.height_channels)
        fg_mask = torch.max(height_labels, dim=1).values > 0.0
        with autocast(enabled=False):
            height_loss = (F.binary_cross_entropy(
                height_preds[fg_mask],
                height_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return 3.0 * height_loss
    
    def get_downsampled_gt_height(self, gt_heights):
        """
        Input:
            gt_heights: [B, N, H, W]
        Output:
            gt_heights: [B*N*h*w, d]
        """
        B, N, H, W = gt_heights.shape
        gt_heights = gt_heights.view(B * N, H // self.downsample_factor,
                                   self.downsample_factor, W // self.downsample_factor,
                                   self.downsample_factor, 1)
        gt_heights = gt_heights.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_heights = gt_heights.view(-1, self.downsample_factor * self.downsample_factor)
        
        gt_heights_tmp = torch.where(gt_heights == 0.0,
                                    1e5 * torch.ones_like(gt_heights),
                                    gt_heights)
        gt_heights = torch.min(gt_heights_tmp, dim=-1).values
        gt_heights = gt_heights.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)
        gt_heights = torch.floor((gt_heights - self.hbound[0]) * self.hbound[2] / (self.hbound[1] - self.hbound[0]))
        gt_heights = torch.where((gt_heights < self.height_channels + 1) & (gt_heights >= 0.0),
                                gt_heights, torch.zeros_like(gt_heights))
        gt_heights = F.one_hot(
            gt_heights.long(), num_classes=self.height_channels + 1).view(-1, self.height_channels + 1)[:, 1:]
        return gt_heights.float()

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:]

        return gt_depths.float()

    def eval_step(self, batch, batch_idx, prefix: str):
        (sweep_imgs, points, points_radar, mats, _, img_metas, _, _) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
        preds = self.model(sweep_imgs, points, points_radar, mats)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            results = self.model.module.get_bboxes(preds, img_metas)
        else:
            results = self.model.get_bboxes(preds, img_metas)
        for i in range(len(results)):
            results[i][0] = results[i][0].tensor.detach().cpu().numpy()
            results[i][1] = results[i][1].detach().cpu().numpy()
            results[i][2] = results[i][2].detach().cpu().numpy()
            results[i].append(img_metas[i])
        return results

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def validation_epoch_end(self, validation_step_outputs):
        all_pred_results = list()
        all_img_metas = list()
        for validation_step_output in validation_step_outputs:
            for i in range(len(validation_step_output)):
                all_pred_results.append(validation_step_output[i][:3])
                all_img_metas.append(validation_step_output[i][3])
        synchronize()
        len_dataset = len(self.val_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:len_dataset]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:len_dataset]
        if get_rank() == 0:
            self.evaluator.evaluate(all_pred_results, all_img_metas)

    def test_epoch_end(self, test_step_outputs):
        all_pred_results = list()
        all_img_metas = list()
        for test_step_output in test_step_outputs:
            for i in range(len(test_step_output)):
                all_pred_results.append(test_step_output[i][:3])
                all_img_metas.append(test_step_output[i][3])
        synchronize()
        # TODO: Change another way.
        dataset_length = len(self.val_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:dataset_length]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:dataset_length]
        if get_rank() == 0:
            self.evaluator.evaluate(all_pred_results, all_img_metas)
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.learning_rate,
                                      betas=(0.9, 0.999),
                                      eps=1e-8,
                                      weight_decay=1e-7)
        scheduler = MultiStepLR(optimizer, milestones=[19, 23], gamma=0.1)
        return [[optimizer], [scheduler]]

    def train_dataloader(self):
        train_dataset = NuscMVDetDataset(
            ida_aug_conf=self.ida_aug_conf,
            classes=self.class_names,
            data_root=self.data_root,
            info_path=os.path.join(self.data_root, f'{self.data_root.split("/")[2]}_12hz_infos_train.pkl'),
            is_train=True,
            use_cbgs=self.data_use_cbgs,
            img_conf=self.img_conf,
            num_sweeps=self.num_sweeps,
            sweep_idxes=self.sweep_idxes,
            key_idxes=self.key_idxes,
            return_depth=self.return_depth,
        )
        from functools import partial

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_depth=self.return_depth),
            sampler=None,
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = NuscMVDetDataset(
            ida_aug_conf=self.ida_aug_conf,
            classes=self.class_names,
            data_root=self.data_root,
            info_path=os.path.join(self.data_root, f'{self.data_root.split("/")[2]}_12hz_infos_val.pkl'),
            is_train=False,
            img_conf=self.img_conf,
            num_sweeps=self.num_sweeps,
            sweep_idxes=self.sweep_idxes,
            key_idxes=self.key_idxes,
            return_depth=False,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            sampler=None,
        )
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        return parent_parser

def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)
    print(args)
    
    # derive default_root_dir from data_root's last folder unless overridden
    last_folder = os.path.basename(os.path.normpath(args.data_root))
    derived_root = os.path.join('./outputs/heightfusion++_radar', last_folder)
    if (not hasattr(args, 'default_root_dir')) or args.default_root_dir == './outputs/heightfusion_radar':
        args.default_root_dir = derived_root
    
    
    
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(args.default_root_dir, 'checkpoints'), filename='{epoch}', every_n_epochs=5, save_last=True, save_top_k=-1)
    model = HeightfusionplusLightningModel(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    if args.evaluate:
        for ckpt_name in os.listdir(args.ckpt_path):
            model_pth = os.path.join(args.ckpt_path, ckpt_name)
            ckpt_tag = os.path.splitext(ckpt_name)[0]
            # 为当前 ckpt 指定独立输出目录
            model.evaluator.output_dir = os.path.join(
                args.default_root_dir, f"eval_{ckpt_tag}"
            )
            trainer.test(model, ckpt_path=model_pth)
    else:
        backup_codebase(os.path.join(args.default_root_dir, 'backup'))
        trainer.fit(model)
        
def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('-e',
                               '--evaluate',
                               dest='evaluate',
                               action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-b', '--batch_size_per_device', type=int, default=2)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path', type=str)
    parser = HeightfusionplusLightningModel.add_model_specific_args(parent_parser)
    parser.add_argument('--data_root', type=str, required=True, help='dataset root path')
    parser.set_defaults(
        profiler='simple',
        deterministic=False,
        max_epochs=100,
        accelerator='ddp',
        num_sanity_val_steps=0,
        gradient_clip_val=5,
        limit_val_batches=0,
        enable_checkpointing=True,
        precision=32,
        default_root_dir='./outputs/heightfusion_radar')
    args = parser.parse_args()
    main(args)

if __name__ == '__main__':
    run_cli()
