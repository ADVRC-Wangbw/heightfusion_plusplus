import torch
import torch.nn as nn
from mmdet3d.ops import spconv as spconv
from mmdet3d.ops.spconv.conv import SubMConv3d

class SparseBatchNorm(nn.Module):
    """自定义稀疏批归一化层，处理 SparseConvTensor"""
    def __init__(self, num_features, eps=0.001, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
    
    def forward(self, x):
        return x.replace_feature(self.bn(x.features))

class SparseReLU(nn.Module):
    """自定义稀疏ReLU层，处理 SparseConvTensor"""
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
    
    def forward(self, x):
        return x.replace_feature(self.relu(x.features))

class NoDownsampleSparseEncoder(nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
        
        # Encoder Layer 1: 16 -> 32 channels
        self.encoder_layer1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16),
            SparseBasicBlock(16, 16),
            # 使用自定义的稀疏层
            SubMConv3d(16, 32, kernel_size=1, bias=False),
            SparseBatchNorm(32, eps=0.001, momentum=0.01),
            SparseReLU(inplace=True)
        )
        
        # Encoder Layer 2: 32 -> 64 channels  
        self.encoder_layer2 = spconv.SparseSequential(
            SparseBasicBlock(32, 32),
            SparseBasicBlock(32, 32),
            # 使用自定义的稀疏层
            SubMConv3d(32, 64, kernel_size=1, bias=False),
            SparseBatchNorm(64, eps=0.001, momentum=0.01),
            SparseReLU(inplace=True)
        )
        
        # Encoder Layer 3: 64 -> 128 channels
        self.encoder_layer3 = spconv.SparseSequential(
            SparseBasicBlock(64, 64),
            SparseBasicBlock(64, 64),
            # 使用自定义的稀疏层
            SubMConv3d(64, 128, kernel_size=1, bias=False),
            SparseBatchNorm(128, eps=0.001, momentum=0.01),
            SparseReLU(inplace=True)
        )
        
        # Encoder Layer 4: 128 -> 128 channels (保持)
        self.encoder_layer4 = spconv.SparseSequential(
            SparseBasicBlock(128, 128),
            SparseBasicBlock(128, 128)
        )
        
        # 所有编码层
        self.encoder_layers = [
            self.encoder_layer1,
            self.encoder_layer2, 
            self.encoder_layer3,
            self.encoder_layer4
        ]
    
    def forward(self, x):
        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)
        return encode_features


class SparseBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 第一个卷积层
        self.conv1 = SubMConv3d(
            in_channels, out_channels, 
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = SparseBatchNorm(out_channels, eps=0.001, momentum=0.01)
        
        # 第二个卷积层
        self.conv2 = SubMConv3d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = SparseBatchNorm(out_channels, eps=0.001, momentum=0.01)
        
        self.relu = SparseReLU(inplace=True)
        
        # 如果输入输出通道数不同，使用1x1卷积调整
        if in_channels != out_channels:
            self.shortcut = spconv.SparseSequential(
                SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False),
                SparseBatchNorm(out_channels, eps=0.001, momentum=0.01)
            )
        else:
            self.shortcut = None
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        
        # 确保稀疏结构一致后再相加
        out = out.replace_feature(out.features + identity.features)
        out = self.relu(out)
        
        return out