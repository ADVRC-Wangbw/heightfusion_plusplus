import numpy as np
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import argparse
warnings.filterwarnings('ignore')

class CustomKITTIEvaluator:
    """
    针对自定义KITTI格式数据集的NuScenes风格评估器
    只评估有预测文件的帧，并过滤不在相机视野或点云视野内的GT框
    """
    
    def __init__(self, gt_folder, pred_folder=None, pred_folders=None, 
                 calib_folder=None, img_width=1242, img_height=375, lidar_range=80.0,
                 classes=['Car', 'Cyclist', 'Pedestrian']):
        """
        初始化评估器
        
        Args:
            gt_folder: 真值标签文件夹路径
            pred_folder: 单个预测标签文件夹路径
            pred_folders: 多个预测标签文件夹路径列表
            calib_folder: 标定文件夹路径（每个帧对应的calib文件）
            img_width: 图像宽度（像素）
            img_height: 图像高度（像素）
            lidar_range: 点云有效范围（米），用于过滤距离过远的GT框
            classes: 要评估的类别列表
        """
        self.gt_folder = Path(gt_folder)
        
        if pred_folder:
            self.pred_folder = Path(pred_folder)
            self.pred_folders = None
        elif pred_folders:
            self.pred_folders = [Path(folder) for folder in pred_folders]
            self.pred_folder = None
        else:
            raise ValueError("必须提供pred_folder或pred_folders中的一个")
            
        self.calib_folder = Path(calib_folder) if calib_folder else None
        self.img_width = img_width
        self.img_height = img_height
        self.lidar_range = lidar_range
        
        self.classes = classes
        
        # NuScenes评估参数 - 为每个类别设置不同的阈值
        self.center_distance_thresholds = {
            'Car': 2.0,      # Car类别的匹配距离阈值
            'Cyclist': 1.5,  # Cyclist类别的匹配距离阈值
            'Pedestrian': 1.0  # Pedestrian类别的匹配距离阈值
        }
        
        # IoU阈值 - 为每个类别设置不同的阈值
        self.iou_thresholds = {
            'Car': 0.25,      # Car类别的IoU阈值
            'Cyclist': 0.2,   # Cyclist类别的IoU阈值
            'Pedestrian': 0.2  # Pedestrian类别的IoU阈值
        }
        
        # 定义NuScenes的5个误差指标的最大允许误差（用于归一化）
        self.max_errors = {
            'ATE': 2.0,      # 平均平移误差最大2米
            'ASE': 0.5,      # 平均尺度误差最大50%
            'AOE': np.pi,    # 平均方向误差最大π弧度
            'AVE': 1.0,      # 平均速度误差最大1m/s
            'AAE': 1.0       # 平均属性误差最大1
        }
        
        # 类别映射到NuScenes类别（用于计算权重）
        self.nuscenes_class_mapping = {
            'Car': 'car',
            'Cyclist': 'bicycle',  # 或'cycle'，根据数据集
            'Pedestrian': 'pedestrian'
        }
        
        # NuScenes中每个类别的权重（用于计算mAP）
        self.class_weights = {
            'Car': 1.0,
            'Cyclist': 2.0,      # 通常小目标权重更高
            'Pedestrian': 2.0
        }
        
        # 定义方向误差计算方式
        self.orientation_metrics = ['yaw']  # 只计算偏航角误差
        
        # 缓存标定数据，避免重复读取
        self.calib_cache = {}
        
    def load_calibration(self, calib_file):
        """
        加载标定文件，返回解析后的矩阵
        
        Returns:
            dict: 包含 P2, R0_rect, Tr_velo_to_cam 等矩阵
        """
        if calib_file in self.calib_cache:
            return self.calib_cache[calib_file]
        
        calib = {}
        with open(calib_file, 'r') as f:
            for line in f:
                if line.startswith('P0:'):
                    data = line.strip().split()[1:]
                    calib['P0'] = np.array([float(x) for x in data]).reshape(3, 4)
                elif line.startswith('P1:'):
                    data = line.strip().split()[1:]
                    calib['P1'] = np.array([float(x) for x in data]).reshape(3, 4)
                elif line.startswith('P2:'):
                    data = line.strip().split()[1:]
                    calib['P2'] = np.array([float(x) for x in data]).reshape(3, 4)
                elif line.startswith('P3:'):
                    data = line.strip().split()[1:]
                    calib['P3'] = np.array([float(x) for x in data]).reshape(3, 4)
                elif line.startswith('R0_rect:'):
                    data = line.strip().split()[1:]
                    calib['R0_rect'] = np.array([float(x) for x in data]).reshape(3, 3)
                elif line.startswith('Tr_velo_to_cam:'):
                    data = line.strip().split()[1:]
                    calib['Tr_velo_to_cam'] = np.array([float(x) for x in data]).reshape(3, 4)
                elif line.startswith('Tr_velo_to_global:'):
                    data = line.strip().split()[1:]
                    calib['Tr_velo_to_global'] = np.array([float(x) for x in data]).reshape(4, 4)
        
        self.calib_cache[calib_file] = calib
        return calib
    
    def get_3d_box_vertices(self, bbox):
        """
        计算3D边界框的8个顶点坐标（在相机坐标系下）
        
        Args:
            bbox: 包含 location (x,y,z), dimensions (h,w,l), rotation_y 的字典
        
        Returns:
            np.array: (8, 3) 顶点坐标，顺序为左下前、左上前后等
        """
        x, y, z = bbox['location']
        h, w, l = bbox['dimensions']  # h: 高度, w: 宽度, l: 长度
        ry = bbox['rotation_y']
        
        # 在物体局部坐标系中，中心位于原点，尺寸为 (l, w, h) 对应的轴为 (x, y, z)
        # 我们定义8个顶点相对于中心的偏移 (dx, dy, dz) 分别对应 (x, y, z) 方向
        # 注意：KITTI中物体坐标：x向前，y向左，z向上？实际相机坐标系中 x右，y下，z前
        # 但物体尺寸: h (高度), w (宽度), l (长度)，分别对应 z轴（向上），y轴（向左），x轴（向前）
        # 所以顶点偏移需要对应:
        # 在相机坐标系中：x 向前，y 向左，z 向上。那么物体的 x 方向是长度，y 方向是宽度，z 方向是高度。
        # 因此，局部坐标偏移 (dx, dy, dz) = (l/2, w/2, h/2) 但要注意正负号。
        # 通常，我们使用如下顺序：
        # 0: (+l/2, +w/2, +h/2)
        # 1: (+l/2, +w/2, -h/2)
        # 2: (+l/2, -w/2, +h/2)
        # 3: (+l/2, -w/2, -h/2)
        # 4: (-l/2, +w/2, +h/2)
        # 5: (-l/2, +w/2, -h/2)
        # 6: (-l/2, -w/2, +h/2)
        # 7: (-l/2, -w/2, -h/2)
        # 其中 x 向前，y 向左，z 向上。
        # 但根据KITTI，ry 是绕 Y 轴（高度轴）旋转，所以旋转矩阵为：
        # R = [[cos(ry), 0, sin(ry)], [0, 1, 0], [-sin(ry), 0, cos(ry)]]
        # 注意：实际KITTI中，旋转矩阵绕 Y 轴（y 轴）旋转，但这里 y 轴是向左（在相机坐标系中），所以公式正确。
        
        l2 = l / 2.0
        w2 = w / 2.0
        h2 = h / 2.0
        
        # 局部坐标
        corners = np.array([
            [ l2,  w2,  h2],   # 0
            [ l2,  w2, -h2],   # 1
            [ l2, -w2,  h2],   # 2
            [ l2, -w2, -h2],   # 3
            [-l2,  w2,  h2],   # 4
            [-l2,  w2, -h2],   # 5
            [-l2, -w2,  h2],   # 6
            [-l2, -w2, -h2]    # 7
        ])
        
        # 旋转矩阵
        R = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        # 旋转后加上中心点
        corners = corners @ R.T + np.array([x, y, z])
        
        return corners
    
    def project_3d_to_2d(self, pts_3d, P):
        """
        将3D点投影到图像平面
        
        Args:
            pts_3d: (N, 3) 点集
            P: (3, 4) 投影矩阵
        
        Returns:
            (N, 2) 图像坐标 (u, v)
        """
        # 转为齐次坐标
        pts_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
        proj = pts_hom @ P.T
        # 归一化
        u = proj[:, 0] / proj[:, 2]
        v = proj[:, 1] / proj[:, 2]
        return np.stack([u, v], axis=1)
    
    def is_in_camera_fov(self, bbox, calib):
        """
        判断3D框是否在相机视野内（至少一个顶点可见）
        """
        corners = self.get_3d_box_vertices(bbox)
        # 使用P2矩阵投影（通常P2对应彩色相机）
        P = calib.get('P2')
        if P is None:
            # 如果没有P2，尝试使用P0
            P = calib.get('P0')
        if P is None:
            print("警告：标定文件中缺少投影矩阵，无法进行相机视野过滤")
            return True
        
        uv = self.project_3d_to_2d(corners, P)
        # 检查是否有顶点落在图像内
        in_image = (uv[:, 0] >= 0) & (uv[:, 0] < self.img_width) & \
                   (uv[:, 1] >= 0) & (uv[:, 1] < self.img_height)
        return np.any(in_image)
    
    def transform_points_cam_to_lidar(self, pts_cam, calib):
        """
        将相机坐标系下的点转换到激光雷达坐标系
        
        公式：pts_lidar = inv(Tr_velo_to_cam) * R0_rect_inv * pts_cam_hom
        其中 Tr_velo_to_cam 是 3x4， R0_rect 是 3x3
        实际上，我们需要将相机点转换到激光雷达坐标：
        pts_lidar = inv(Tr_velo_to_cam) * inv(R0_rect) * pts_cam_hom
        注意：R0_rect 是旋转矩阵，逆等于转置
        """
        Tr = calib.get('Tr_velo_to_cam')
        R0 = calib.get('R0_rect')
        if Tr is None or R0 is None:
            print("警告：标定文件中缺少 Tr_velo_to_cam 或 R0_rect，无法进行点云视野过滤")
            return pts_cam
        
        # 构建 4x4 变换矩阵 cam -> lidar
        # 首先，从相机坐标系转换到修正前的相机坐标系（即原始相机坐标系）：乘以 R0_rect^T
        # 然后从修正前的相机坐标系转换到激光雷达坐标系：乘以 Tr_velo_to_cam 的逆
        # 合并：M_cam_to_lidar = inv(Tr_velo_to_cam) * inv(R0_rect)
        # 但注意，R0_rect 是 3x3，需要扩展为 4x4，Tr_velo_to_cam 是 3x4，扩展为 4x4
        
        # 构建 R0_rect 的 4x4 矩阵
        R0_4x4 = np.eye(4)
        R0_4x4[:3, :3] = R0
        # Tr_velo_to_cam 是 3x4，扩展为 4x4
        Tr_velo_to_cam_4x4 = np.eye(4)
        Tr_velo_to_cam_4x4[:3, :] = Tr
        # 从相机到激光雷达的变换 = inv(Tr_velo_to_cam) * inv(R0)
        M_cam_to_lidar = np.linalg.inv(Tr_velo_to_cam_4x4) @ np.linalg.inv(R0_4x4)
        
        # 添加齐次坐标
        pts_hom = np.hstack((pts_cam, np.ones((pts_cam.shape[0], 1))))
        pts_lidar_hom = pts_hom @ M_cam_to_lidar.T
        return pts_lidar_hom[:, :3]
    
    def is_in_lidar_fov(self, bbox, calib):
        """
        判断3D框是否在点云有效范围内（至少有一个顶点在点云FOV内）
        简化：检查框的中心点是否在距离阈值内（且水平角度在合理范围内）
        """
        # 获取框中心点（相机坐标）
        center = np.array(bbox['location']).reshape(1, 3)
        # 转换到激光雷达坐标
        center_lidar = self.transform_points_cam_to_lidar(center, calib)
        # 计算距离
        dist = np.linalg.norm(center_lidar[0])
        # 如果距离超过阈值，则不在视野内
        if dist > self.lidar_range:
            return False
        
        # 可选：检查水平角度是否在 ±90° 内（即前方180度）
        # 激光雷达坐标系中，x 向前，y 向左
        x, y, _ = center_lidar[0]
        angle = np.arctan2(y, x)
        if abs(angle) > np.pi/2:
            return False
        
        return True
    
    def filter_gt_box(self, bbox, calib):
        """
        过滤GT框：必须同时满足在相机视野和点云视野内
        """
        if not self.is_in_camera_fov(bbox, calib):
            return False
        if not self.is_in_lidar_fov(bbox, calib):
            return False
        return True
    
    def parse_line(self, line, is_gt=False):
        """
        解析单行标签数据
        """
        parts = line.strip().split()
        
        if not parts or len(parts) < 14:
            return None
            
        try:
            if is_gt:
                # GT格式: type ? ? ? ? ? ? ? h w l x y z ry (15个字段)
                # 有些可能有16个字段，我们取前15个
                if len(parts) >= 15:
                    obj = {
                        'type': parts[0],
                        'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],  # h, w, l
                        'location': [float(parts[11]), float(parts[12]), float(parts[13])],  # x, y, z
                        'rotation_y': float(parts[14]) if len(parts) > 14 else 0.0,
                        'score': 1.0
                    }
                    # 尝试解析速度和属性（如果存在）
                    if len(parts) >= 17:
                        obj['velocity'] = [float(parts[15]), float(parts[16])]  # vx, vy
                    if len(parts) >= 18:
                        obj['attribute'] = float(parts[17])  # 属性得分
                    return obj
            else:
                # 预测格式: type ? ? ? bbox[4] h w l x y z ry score (16个字段)
                if len(parts) >= 16:
                    obj = {
                        'type': parts[0],
                        'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],  # h, w, l
                        'location': [float(parts[11]), float(parts[12]), float(parts[13])],  # x, y, z
                        'rotation_y': float(parts[14]) if len(parts) > 14 else 0.0,
                        'score': float(parts[15]) if len(parts) > 15 else 0.5
                    }
                    # 尝试解析速度和属性（如果存在）
                    if len(parts) >= 18:
                        obj['velocity'] = [float(parts[16]), float(parts[17])]
                    if len(parts) >= 19:
                        obj['attribute'] = float(parts[18])
                    return obj
        except (ValueError, IndexError) as e:
            print(f"解析错误: {line}, 错误: {e}")
            return None
        
        return None
    
    def load_data(self, pred_folder=None):
        """
        只加载有预测文件的帧的GT和预测数据，并对GT应用视野过滤
        """
        if pred_folder is None:
            pred_folder = self.pred_folder
            
        gt_data = {}
        pred_data = {}
        
        # 获取所有预测文件
        pred_files = sorted(pred_folder.glob("*.txt"))
        
        print(f"找到 {len(pred_files)} 个预测文件")
        print(f"GT文件夹中有 {len(list(self.gt_folder.glob('*.txt')))} 个GT文件")
        
        valid_frames = []
        
        for pred_file in tqdm(pred_files, desc="加载数据"):
            frame_id = pred_file.stem
            
            # 检查对应的GT文件是否存在
            gt_file = self.gt_folder / f"{frame_id}.txt"
            if not gt_file.exists():
                continue
            
            # 加载该帧的标定文件
            if self.calib_folder is not None:
                calib_file = self.calib_folder / f"{frame_id}.txt"
                if not calib_file.exists():
                    print(f"警告：帧 {frame_id} 缺少标定文件，跳过GT过滤")
                    calib = None
                else:
                    calib = self.load_calibration(calib_file)
            else:
                calib = None
            
            # 加载GT并过滤
            gt_boxes = []
            with open(gt_file, 'r') as f:
                for line in f:
                    parsed = self.parse_line(line, is_gt=True)
                    if parsed and parsed['type'] in self.classes:
                        # 应用视野过滤（如果有标定）
                        if calib is not None:
                            if self.filter_gt_box(parsed, calib):
                                gt_boxes.append(parsed)
                        else:
                            # 没有标定时，保留所有框（但给出警告）
                            if len(gt_boxes) == 0:  # 仅警告一次
                                print(f"警告：帧 {frame_id} 缺少标定文件，GT未过滤")
                            gt_boxes.append(parsed)
            
            # 加载预测（预测框不需要过滤）
            pred_boxes = []
            with open(pred_file, 'r') as f:
                for line in f:
                    parsed = self.parse_line(line, is_gt=False)
                    if parsed and parsed['type'] in self.classes:
                        pred_boxes.append(parsed)
            
            # 只有存在至少一个GT或预测时才保留该帧
            if gt_boxes or pred_boxes:
                valid_frames.append(frame_id)
                gt_data[frame_id] = gt_boxes
                pred_data[frame_id] = pred_boxes
        
        print(f"\n有效帧数（既有GT又有预测）: {len(valid_frames)}")
        
        # 统计信息
        total_gts = sum(len(gt_data[f]) for f in valid_frames)
        total_preds = sum(len(pred_data[f]) for f in valid_frames)
        
        print(f"有效帧中的GT总数（过滤后）: {total_gts}")
        print(f"预测总数: {total_preds}")
        
        # 按类别统计
        print("\n按类别统计:")
        for cls in self.classes:
            cls_gts = sum(len([gt for gt in gt_data[f] if gt['type'] == cls]) for f in valid_frames)
            cls_preds = sum(len([pred for pred in pred_data[f] if pred['type'] == cls]) for f in valid_frames)
            print(f"  {cls}: GT={cls_gts}, 预测={cls_preds}")
        
        return gt_data, pred_data, valid_frames
    
    def calculate_center_distance(self, box1, box2):
        """
        计算两个3D框中心点之间的距离
        """
        loc1 = np.array(box1['location'])
        loc2 = np.array(box2['location'])
        return np.linalg.norm(loc1 - loc2)
    
    def calculate_scale_error(self, box1, box2):
        """
        计算尺度误差（ASE: Average Scale Error）
        返回三个维度上相对误差的平均值
        """
        h1, w1, l1 = box1['dimensions']
        h2, w2, l2 = box2['dimensions']
        
        # 计算每个维度的相对误差
        h_err = abs(h1 - h2) / max(h1, 0.1)  # 防止除以0
        w_err = abs(w1 - w2) / max(w1, 0.1)
        l_err = abs(l1 - l2) / max(l1, 0.1)
        
        # 返回平均相对误差
        return (h_err + w_err + l_err) / 3.0
    
    def calculate_orientation_error(self, box1, box2):
        """
        计算方向误差（AOE: Average Orientation Error）
        计算偏航角（yaw）的差异，考虑周期性
        """
        yaw1 = box1.get('rotation_y', 0.0)
        yaw2 = box2.get('rotation_y', 0.0)
        
        # 计算角度差异，考虑2π周期性
        diff = abs(yaw1 - yaw2)
        diff = min(diff, 2 * np.pi - diff)  # 取最小差异
        
        return diff
    
    def calculate_velocity_error(self, box1, box2):
        """
        计算速度误差（AVE: Average Velocity Error）
        如果数据中包含速度信息
        """
        v1 = box1.get('velocity', [0.0, 0.0])
        v2 = box2.get('velocity', [0.0, 0.0])
        
        # 计算速度向量差的模
        v1_arr = np.array(v1[:2])  # 取前两个分量
        v2_arr = np.array(v2[:2])
        
        if len(v1_arr) > 0 and len(v2_arr) > 0:
            return np.linalg.norm(v1_arr - v2_arr)
        else:
            return 0.0  # 如果没有速度信息，返回0
    
    def calculate_attribute_error(self, box1, box2):
        """
        计算属性误差（AAE: Average Attribute Error）
        如果数据中包含属性信息
        """
        attr1 = box1.get('attribute', 0.0)
        attr2 = box2.get('attribute', 0.0)
        
        return abs(attr1 - attr2)
    
    def calculate_3d_iou(self, box1, box2):
        """
        简化版3D IoU计算
        """
        # 计算两个框的体积
        h1, w1, l1 = box1['dimensions']
        h2, w2, l2 = box2['dimensions']
        
        vol1 = h1 * w1 * l1
        vol2 = h2 * w2 * l2
        
        # 如果中心距离太远，直接返回0
        center_dist = self.calculate_center_distance(box1, box2)
        max_diag = np.sqrt(max(h1**2 + w1**2 + l1**2, h2**2 + w2**2 + l2**2))
        
        if center_dist > max_diag:
            return 0.0
        
        # 简单估计重叠体积
        # 假设框是轴对齐的（简化计算）
        x1, y1, z1 = box1['location']
        x2, y2, z2 = box2['location']
        
        # 计算每个维度的重叠
        def overlap_1d(center1, center2, size1, size2):
            left1 = center1 - size1 / 2
            right1 = center1 + size1 / 2
            left2 = center2 - size2 / 2
            right2 = center2 + size2 / 2
            
            overlap = max(0, min(right1, right2) - max(left1, left2))
            return overlap
        
        # 注意：KITTI坐标系中，x向前，y向左，z向上
        # 但这里我们按世界坐标系处理
        overlap_x = overlap_1d(x1, x2, l1, l2)  # x方向对应长度
        overlap_y = overlap_1d(y1, y2, h1, h2)  # y方向对应高度（注意：KITTI中y是高度）
        overlap_z = overlap_1d(z1, z2, w1, w2)  # z方向对应宽度
        
        intersection = overlap_x * overlap_y * overlap_z
        union = vol1 + vol2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_detections(self, gt_boxes, pred_boxes, class_name):
        """
        匹配检测框
        """
        if not gt_boxes or not pred_boxes:
            return [], [], []
        
        # 按置信度排序预测框
        sorted_preds = sorted(pred_boxes, key=lambda x: x['score'], reverse=True)
        
        matches = []
        matched_gts = set()
        matched_preds = set()
        
        # 为每个预测框寻找最佳匹配的GT框
        for pred_idx, pred_box in enumerate(sorted_preds):
            best_iou = 0
            best_gt_idx = -1
            best_distance = float('inf')
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gts:
                    continue
                
                # 计算中心距离
                distance = self.calculate_center_distance(gt_box, pred_box)
                
                # 如果距离超过阈值，跳过
                if distance > self.center_distance_thresholds.get(class_name, 2.0):
                    continue
                
                # 计算IoU
                iou = self.calculate_3d_iou(gt_box, pred_box)
                
                # 如果IoU大于阈值且是最好的匹配
                if iou > self.iou_thresholds.get(class_name, 0.25) and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    best_distance = distance
            
            if best_gt_idx != -1:
                # 计算所有NuScenes误差指标
                gt_box = gt_boxes[best_gt_idx]
                scale_error = self.calculate_scale_error(gt_box, pred_box)
                orientation_error = self.calculate_orientation_error(gt_box, pred_box)
                velocity_error = self.calculate_velocity_error(gt_box, pred_box)
                attribute_error = self.calculate_attribute_error(gt_box, pred_box)
                
                matches.append({
                    'gt_idx': best_gt_idx,
                    'pred_idx': pred_idx,  # 注意：这是在sorted_preds中的索引
                    'iou': best_iou,
                    'distance': best_distance,
                    'scale_error': scale_error,
                    'orientation_error': orientation_error,
                    'velocity_error': velocity_error,
                    'attribute_error': attribute_error
                })
                matched_gts.add(best_gt_idx)
                matched_preds.add(pred_idx)
        
        # 获取未匹配的GT和预测
        unmatched_gts = [i for i in range(len(gt_boxes)) if i not in matched_gts]
        unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_preds]
        
        return matches, unmatched_gts, unmatched_preds
    
    def calculate_ap(self, precisions, recalls):
        """
        计算平均精度（AP）
        """
        # 确保从召回率0开始
        mrec = np.concatenate(([0.], recalls, [1.]))
        mpre = np.concatenate(([0.], precisions, [0.]))
        
        # 使精度单调递减
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        # 找到召回率变化的点
        i = np.where(mrec[1:] != mrec[:-1])[0]
        
        # 计算AP
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        
        return ap
    
    def evaluate_class(self, gt_data, pred_data, valid_frames, class_name):
        """
        评估单个类别
        """
        all_scores = []
        all_tp = []  # 1表示真正例，0表示假正例
        all_distances = []  # 只存储匹配的预测框的距离
        all_ious = []  # 只存储匹配的预测框的IoU
        all_scale_errors = []  # 尺度误差
        all_orientation_errors = []  # 方向误差
        all_velocity_errors = []  # 速度误差
        all_attribute_errors = []  # 属性误差
        
        # 收集所有匹配信息
        for frame_id in valid_frames:
            # 获取当前类别的GT和预测
            frame_gts = gt_data.get(frame_id, [])
            frame_preds = pred_data.get(frame_id, [])
            
            # 只处理当前类别的框
            frame_gts = [gt for gt in frame_gts if gt['type'] == class_name]
            frame_preds = [pred for pred in frame_preds if pred['type'] == class_name]
            
            if not frame_gts and not frame_preds:
                continue
            
            # 匹配检测框
            matches, unmatched_gts, unmatched_preds = self.match_detections(
                frame_gts, frame_preds, class_name
            )
            
            # 处理匹配的预测（真正例）
            for match in matches:
                pred_box = frame_preds[match['pred_idx']]
                all_scores.append(pred_box['score'])
                all_tp.append(1)
                all_distances.append(match['distance'])
                all_ious.append(match['iou'])
                all_scale_errors.append(match['scale_error'])
                all_orientation_errors.append(match['orientation_error'])
                all_velocity_errors.append(match['velocity_error'])
                all_attribute_errors.append(match['attribute_error'])
            
            # 处理未匹配的预测（假正例）
            for pred_idx in unmatched_preds:
                pred_box = frame_preds[pred_idx]
                all_scores.append(pred_box['score'])
                all_tp.append(0)
        
        # 如果没有预测，返回0
        if not all_scores:
            return {
                'AP': 0.0,
                'ATE': 0.0,
                'ASE': 0.0,
                'AOE': 0.0,
                'AVE': 0.0,
                'AAE': 0.0,
                'mean_iou': 0.0,
                'recall': 0.0,
                'precision': 0.0,
                'total_gts': 0,
                'total_preds': 0,
                'total_matches': 0
            }
        
        # 计算该类别的GT总数（只计算有效帧）
        total_gts = sum(
            len([gt for gt in gt_data[f] if gt['type'] == class_name]) 
            for f in valid_frames
        )
        
        # 按置信度排序
        sorted_indices = np.argsort(-np.array(all_scores))
        sorted_scores = np.array(all_scores)[sorted_indices]
        sorted_tp = np.array(all_tp)[sorted_indices]
        
        # 计算累积TP和FP
        tp_cumsum = np.cumsum(sorted_tp)
        fp_cumsum = np.cumsum(1 - sorted_tp)
        
        # 计算精度和召回率
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        recalls = tp_cumsum / max(total_gts, 1)
        
        # 计算AP
        ap = self.calculate_ap(precisions, recalls)
        
        # 计算各种误差指标
        metrics = {}
        
        # 平均平移误差（ATE）
        if all_distances:
            metrics['ATE'] = np.mean(all_distances)
        else:
            metrics['ATE'] = 0.0
        
        # 平均尺度误差（ASE）
        if all_scale_errors:
            metrics['ASE'] = np.mean(all_scale_errors)
        else:
            metrics['ASE'] = 0.0
        
        # 平均方向误差（AOE）
        if all_orientation_errors:
            metrics['AOE'] = np.mean(all_orientation_errors)
        else:
            metrics['AOE'] = 0.0
        
        # 平均速度误差（AVE）
        if all_velocity_errors:
            metrics['AVE'] = np.mean(all_velocity_errors)
        else:
            metrics['AVE'] = 0.0
        
        # 平均属性误差（AAE）
        if all_attribute_errors:
            metrics['AAE'] = np.mean(all_attribute_errors)
        else:
            metrics['AAE'] = 0.0
        
        # 计算平均IoU
        if all_ious:
            metrics['mean_iou'] = np.mean(all_ious)
        else:
            metrics['mean_iou'] = 0.0
        
        # 计算最大召回率时的精度
        max_recall = recalls[-1] if len(recalls) > 0 else 0.0
        max_precision = precisions[-1] if len(precisions) > 0 else 0.0
        
        # 归一化误差指标（用于NDS计算）
        metrics['norm_ATE'] = min(metrics['ATE'] / self.max_errors['ATE'], 1.0)
        metrics['norm_ASE'] = min(metrics['ASE'] / self.max_errors['ASE'], 1.0)
        metrics['norm_AOE'] = min(metrics['AOE'] / self.max_errors['AOE'], 1.0)
        metrics['norm_AVE'] = min(metrics['AVE'] / self.max_errors['AVE'], 1.0)
        metrics['norm_AAE'] = min(metrics['AAE'] / self.max_errors['AAE'], 1.0)
        
        result = {
            'AP': ap,
            'ATE': metrics['ATE'],
            'ASE': metrics['ASE'],
            'AOE': metrics['AOE'],
            'AVE': metrics['AVE'],
            'AAE': metrics['AAE'],
            'norm_ATE': metrics['norm_ATE'],
            'norm_ASE': metrics['norm_ASE'],
            'norm_AOE': metrics['norm_AOE'],
            'norm_AVE': metrics['norm_AVE'],
            'norm_AAE': metrics['norm_AAE'],
            'mean_iou': metrics['mean_iou'],
            'recall': max_recall,
            'precision': max_precision,
            'total_gts': total_gts,
            'total_preds': len(all_scores),
            'total_matches': len(all_distances)  # 匹配数量等于距离列表长度
        }
        
        return result
    
    def calculate_nds(self, metrics):
        """
        计算NDS（NuScenes Detection Score）
        
        公式: NDS = 0.1 * (5 * mAP + ∑(1 - min(1, error/threshold)))
        其中error是ATE, ASE, AOE, AVE, AAE
        """
        # 计算mAP（使用类别权重）
        aps = [metrics[cls]['AP'] * self.class_weights.get(cls, 1.0) 
               for cls in self.classes if cls in metrics]
        weights = [self.class_weights.get(cls, 1.0) for cls in self.classes if cls in metrics]
        
        if sum(weights) > 0:
            mAP = sum(aps) / sum(weights)
        else:
            mAP = 0.0
        
        # 计算每个归一化误差指标的平均值
        norm_errors = []
        
        # 收集所有类别的归一化误差
        for cls in self.classes:
            if cls in metrics:
                norm_errors.append({
                    'ATE': metrics[cls]['norm_ATE'],
                    'ASE': metrics[cls]['norm_ASE'],
                    'AOE': metrics[cls]['norm_AOE'],
                    'AVE': metrics[cls]['norm_AVE'],
                    'AAE': metrics[cls]['norm_AAE']
                })
        
        # 计算平均归一化误差
        if norm_errors:
            mATE = np.mean([err['ATE'] for err in norm_errors])
            mASE = np.mean([err['ASE'] for err in norm_errors])
            mAOE = np.mean([err['AOE'] for err in norm_errors])
            mAVE = np.mean([err['AVE'] for err in norm_errors])
            mAAE = np.mean([err['AAE'] for err in norm_errors])
        else:
            mATE = mASE = mAOE = mAVE = mAAE = 1.0  # 如果没有匹配，误差为最大值
        
        # 计算NDS
        error_sum = (1 - mATE) + (1 - mASE) + (1 - mAOE) + (1 - mAVE) + (1 - mAAE)
        nds = 0.1 * (5 * mAP + error_sum)
        
        # 返回NDS和各个指标
        return nds, mAP, {
            'mATE': mATE * self.max_errors['ATE'],  # 转换回原始尺度
            'mASE': mASE * self.max_errors['ASE'],
            'mAOE': mAOE * self.max_errors['AOE'],
            'mAVE': mAVE * self.max_errors['AVE'],
            'mAAE': mAAE * self.max_errors['AAE'],
            'norm_mATE': mATE,
            'norm_mASE': mASE,
            'norm_mAOE': mAOE,
            'norm_mAVE': mAVE,
            'norm_mAAE': mAAE
        }
    
    def evaluate(self, pred_folder=None):
        """
        执行完整评估
        
        Args:
            pred_folder: 可选的预测文件夹路径，如果不指定则使用self.pred_folder
        """
        print("开始评估...")
        print(f"计算指标: AP, ATE, ASE, AOE, AVE, AAE, NDS")
        
        if pred_folder is not None:
            use_folder = pred_folder
        else:
            use_folder = self.pred_folder
        
        # 加载数据（只加载有预测的帧）
        gt_data, pred_data, valid_frames = self.load_data(use_folder)
        
        if not valid_frames:
            print(f"错误：没有有效的帧可以评估！预测文件夹: {use_folder}")
            return None
        
        # 评估每个类别
        metrics = {}
        for class_name in self.classes:
            print(f"\n评估类别: {class_name}")
            class_metrics = self.evaluate_class(gt_data, pred_data, valid_frames, class_name)
            metrics[class_name] = class_metrics
            
            print(f"  AP: {class_metrics['AP']:.4f}")
            print(f"  召回率: {class_metrics['recall']:.4f}")
            print(f"  精确率: {class_metrics['precision']:.4f}")
            print(f"  ATE: {class_metrics['ATE']:.4f} m")
            print(f"  ASE: {class_metrics['ASE']:.4f}")
            print(f"  AOE: {class_metrics['AOE']:.4f} rad")
            print(f"  AVE: {class_metrics['AVE']:.4f} m/s")
            print(f"  AAE: {class_metrics['AAE']:.4f}")
            print(f"  平均IoU: {class_metrics['mean_iou']:.4f}")
            print(f"  GT数量: {class_metrics['total_gts']}")
            print(f"  预测数量: {class_metrics['total_preds']}")
            print(f"  匹配数量: {class_metrics['total_matches']}")
        
        # 计算NDS和NuScenes误差指标
        nds, mAP, nuscenes_metrics = self.calculate_nds(metrics)
        
        # 汇总结果
        result = {
            'metrics_per_class': metrics,
            'mAP': mAP,
            'NDS': nds,
            'nuscenes_metrics': nuscenes_metrics,
            'valid_frames': len(valid_frames),
            'summary': {
                'ATE': np.mean([metrics[cls]['ATE'] for cls in self.classes if cls in metrics]),
                'ASE': np.mean([metrics[cls]['ASE'] for cls in self.classes if cls in metrics]),
                'AOE': np.mean([metrics[cls]['AOE'] for cls in self.classes if cls in metrics]),
                'AVE': np.mean([metrics[cls]['AVE'] for cls in self.classes if cls in metrics]),
                'AAE': np.mean([metrics[cls]['AAE'] for cls in self.classes if cls in metrics]),
                'mean_iou': np.mean([metrics[cls]['mean_iou'] for cls in self.classes if cls in metrics]),
                'total_gts': sum(metrics[cls]['total_gts'] for cls in self.classes if cls in metrics),
                'total_preds': sum(metrics[cls]['total_preds'] for cls in self.classes if cls in metrics),
                'total_matches': sum(metrics[cls]['total_matches'] for cls in self.classes if cls in metrics)
            }
        }
        
        # 打印汇总结果
        print("\n" + "="*80)
        print("评估结果汇总:")
        print("="*80)
        print(f"有效帧数: {result['valid_frames']}")
        print(f"mAP: {mAP:.4f}")
        print(f"NDS: {nds:.4f}")
        print(f"\nNuScenes误差指标:")
        print(f"  mATE: {nuscenes_metrics['mATE']:.4f} m")
        print(f"  mASE: {nuscenes_metrics['mASE']:.4f}")
        print(f"  mAOE: {nuscenes_metrics['mAOE']:.4f} rad")
        print(f"  mAVE: {nuscenes_metrics['mAVE']:.4f} m/s")
        print(f"  mAAE: {nuscenes_metrics['mAAE']:.4f}")
        print(f"\n归一化NuScenes误差指标:")
        print(f"  norm_mATE: {nuscenes_metrics['norm_mATE']:.4f}")
        print(f"  norm_mASE: {nuscenes_metrics['norm_mASE']:.4f}")
        print(f"  norm_mAOE: {nuscenes_metrics['norm_mAOE']:.4f}")
        print(f"  norm_mAVE: {nuscenes_metrics['norm_mAVE']:.4f}")
        print(f"  norm_mAAE: {nuscenes_metrics['norm_mAAE']:.4f}")
        print(f"\n其他指标:")
        print(f"平均ATE: {result['summary']['ATE']:.4f} m")
        print(f"平均ASE: {result['summary']['ASE']:.4f}")
        print(f"平均AOE: {result['summary']['AOE']:.4f} rad")
        print(f"平均AVE: {result['summary']['AVE']:.4f} m/s")
        print(f"平均AAE: {result['summary']['AAE']:.4f}")
        print(f"平均IoU: {result['summary']['mean_iou']:.4f}")
        print(f"总GT数: {result['summary']['total_gts']}")
        print(f"总预测数: {result['summary']['total_preds']}")
        print(f"总匹配数: {result['summary']['total_matches']}")
        print(f"匹配率: {result['summary']['total_matches']/max(result['summary']['total_preds'],1):.2%}")
        
        return result, use_folder
    
    def evaluate_multiple_folders(self):
        """
        评估多个预测文件夹，返回mAP最高的文件夹名称
        """
        if not self.pred_folders:
            print("错误：没有提供多个预测文件夹")
            return None
        
        results = {}
        best_mAP = -1
        best_folder = None
        best_result = None
        
        print(f"开始评估 {len(self.pred_folders)} 个预测文件夹...")
        print("="*80)
        
        for i, pred_folder in enumerate(self.pred_folders):
            print(f"\n评估文件夹 {i+1}/{len(self.pred_folders)}: {pred_folder}")
            print("-"*40)
            
            try:
                result, folder_name = self.evaluate(pred_folder)
                if result is not None:
                    results[str(pred_folder)] = {
                        'result': result,
                        'mAP': result['mAP'],
                        'NDS': result['NDS']
                    }
                    
                    # 更新最佳结果
                    if result['mAP'] > best_mAP:
                        best_mAP = result['mAP']
                        best_folder = str(pred_folder)
                        best_result = result
                        
                    print(f"当前文件夹 mAP: {result['mAP']:.4f}")
                    print(f"当前最佳 mAP: {best_mAP:.4f} (来自: {best_folder})")
                    
            except Exception as e:
                print(f"评估文件夹 {pred_folder} 时出错: {e}")
                continue
        
        # 打印所有文件夹的结果
        print("\n" + "="*80)
        print("所有预测文件夹评估结果:")
        print("="*80)
        
        # 按mAP排序
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mAP'], reverse=True)
        
        for i, (folder, data) in enumerate(sorted_results):
            print(f"{i+1}. {folder}")
            print(f"   mAP: {data['mAP']:.4f}, NDS: {data['NDS']:.4f}")
        
        print("\n" + "="*80)
        print(f"最佳预测文件夹: {best_folder}")
        print(f"最佳 mAP: {best_mAP:.4f}")
        print("="*80)
        
        return best_folder, best_mAP, best_result, results
    
    def visualize_results(self, result, output_dir="results", folder_name=None):
        """
        可视化评估结果
        
        Args:
            result: 评估结果
            output_dir: 输出目录
            folder_name: 文件夹名称，用于创建子目录
        """
        if folder_name:
            # 使用文件夹名称创建子目录
            safe_folder_name = folder_name.replace('/', '_').replace('\\', '_')
            output_dir = Path(output_dir) / safe_folder_name
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建柱状图
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # 获取类别
        classes = [cls for cls in self.classes if cls in result['metrics_per_class']]
        
        if not classes:
            print("没有可用的类别数据进行可视化")
            return
        
        x = np.arange(len(classes))
        width = 0.35
        
        # 1. AP
        aps = [result['metrics_per_class'][cls]['AP'] for cls in classes]
        axes[0, 0].bar(classes, aps, color='skyblue')
        axes[0, 0].set_title('AP per Class')
        axes[0, 0].set_ylabel('AP')
        axes[0, 0].set_ylim(0, 1)
        
        # 2. 召回率和精确率
        recalls = [result['metrics_per_class'][cls]['recall'] for cls in classes]
        precisions = [result['metrics_per_class'][cls]['precision'] for cls in classes]
        
        axes[0, 1].bar(x - width/2, recalls, width, label='Recall', color='lightgreen')
        axes[0, 1].bar(x + width/2, precisions, width, label='Precision', color='lightcoral')
        axes[0, 1].set_title('Recall and Precision per Class')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(classes)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)
        
        # 3. ATE
        ates = [result['metrics_per_class'][cls]['ATE'] for cls in classes]
        axes[0, 2].bar(classes, ates, color='orange')
        axes[0, 2].set_title('ATE per Class')
        axes[0, 2].set_ylabel('ATE (m)')
        
        # 4. ASE
        ases = [result['metrics_per_class'][cls]['ASE'] for cls in classes]
        axes[1, 0].bar(classes, ases, color='red')
        axes[1, 0].set_title('ASE per Class')
        axes[1, 0].set_ylabel('ASE')
        
        # 5. AOE
        aoes = [result['metrics_per_class'][cls]['AOE'] for cls in classes]
        axes[1, 1].bar(classes, aoes, color='purple')
        axes[1, 1].set_title('AOE per Class')
        axes[1, 1].set_ylabel('AOE (rad)')
        
        # 6. 平均IoU
        mean_ious = [result['metrics_per_class'][cls]['mean_iou'] for cls in classes]
        axes[1, 2].bar(classes, mean_ious, color='brown')
        axes[1, 2].set_title('Mean IoU per Class')
        axes[1, 2].set_ylabel('Mean IoU')
        axes[1, 2].set_ylim(0, 1)
        
        # 7. 检测数量
        gt_counts = [result['metrics_per_class'][cls]['total_gts'] for cls in classes]
        pred_counts = [result['metrics_per_class'][cls]['total_preds'] for cls in classes]
        match_counts = [result['metrics_per_class'][cls]['total_matches'] for cls in classes]
        
        axes[2, 0].bar(x - width, gt_counts, width, label='GT', color='gray')
        axes[2, 0].bar(x, pred_counts, width, label='Predictions', color='orange')
        axes[2, 0].bar(x + width, match_counts, width, label='Matches', color='green')
        axes[2, 0].set_title('Detection Counts per Class')
        axes[2, 0].set_xticks(x)
        axes[2, 0].set_xticklabels(classes)
        axes[2, 0].legend()
        
        # 8. mAP和NDS
        axes[2, 1].bar(['mAP', 'NDS'], [result['mAP'], result['NDS']], 
                      color=['darkblue', 'darkgreen'])
        axes[2, 1].set_title('Overall Metrics')
        axes[2, 1].set_ylim(0, 1)
        
        # 9. NuScenes误差指标
        nuscenes_metrics = result['nuscenes_metrics']
        ns_metrics = ['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']
        ns_values = [nuscenes_metrics['mATE'], nuscenes_metrics['mASE'], 
                    nuscenes_metrics['mAOE'], nuscenes_metrics['mAVE'], 
                    nuscenes_metrics['mAAE']]
        axes[2, 2].bar(ns_metrics, ns_values, color=['blue', 'red', 'purple', 'orange', 'green'])
        axes[2, 2].set_title('NuScenes Error Metrics')
        axes[2, 2].set_ylabel('Error Value')
        
        plt.suptitle(f'3D目标检测评估结果 - {folder_name if folder_name else "结果"}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(f"{output_dir}/evaluation_results.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存结果到文本文件
        with open(f"{output_dir}/evaluation_results.txt", 'w') as f:
            f.write("="*80 + "\n")
            f.write("3D目标检测评估结果（NuScenes风格）\n")
            if folder_name:
                f.write(f"预测文件夹: {folder_name}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"有效帧数: {result['valid_frames']}\n")
            f.write(f"Overall mAP: {result['mAP']:.4f}\n")
            f.write(f"NDS: {result['NDS']:.4f}\n\n")
            
            f.write("NuScenes误差指标:\n")
            f.write("-"*40 + "\n")
            f.write(f"  mATE: {nuscenes_metrics['mATE']:.4f} m\n")
            f.write(f"  mASE: {nuscenes_metrics['mASE']:.4f}\n")
            f.write(f"  mAOE: {nuscenes_metrics['mAOE']:.4f} rad\n")
            f.write(f"  mAVE: {nuscenes_metrics['mAVE']:.4f} m/s\n")
            f.write(f"  mAAE: {nuscenes_metrics['mAAE']:.4f}\n\n")
            
            f.write("归一化NuScenes误差指标:\n")
            f.write("-"*40 + "\n")
            f.write(f"  norm_mATE: {nuscenes_metrics['norm_mATE']:.4f}\n")
            f.write(f"  norm_mASE: {nuscenes_metrics['norm_mASE']:.4f}\n")
            f.write(f"  norm_mAOE: {nuscenes_metrics['norm_mAOE']:.4f}\n")
            f.write(f"  norm_mAVE: {nuscenes_metrics['norm_mAVE']:.4f}\n")
            f.write(f"  norm_mAAE: {nuscenes_metrics['norm_mAAE']:.4f}\n\n")
            
            f.write("Per-class metrics:\n")
            f.write("-"*80 + "\n")
            for cls in classes:
                metrics_data = result['metrics_per_class'][cls]
                f.write(f"\n{cls}:\n")
                f.write(f"  AP: {metrics_data['AP']:.4f}\n")
                f.write(f"  Recall: {metrics_data['recall']:.4f}\n")
                f.write(f"  Precision: {metrics_data['precision']:.4f}\n")
                f.write(f"  ATE: {metrics_data['ATE']:.4f} m\n")
                f.write(f"  ASE: {metrics_data['ASE']:.4f}\n")
                f.write(f"  AOE: {metrics_data['AOE']:.4f} rad\n")
                f.write(f"  AVE: {metrics_data['AVE']:.4f} m/s\n")
                f.write(f"  AAE: {metrics_data['AAE']:.4f}\n")
                f.write(f"  Mean IoU: {metrics_data['mean_iou']:.4f}\n")
                f.write(f"  GT数量: {metrics_data['total_gts']}\n")
                f.write(f"  预测数量: {metrics_data['total_preds']}\n")
                f.write(f"  匹配数量: {metrics_data['total_matches']}\n")
        
        print(f"\n结果已保存到 {output_dir}/ 目录")


def main():
    """
    执行完整的3D目标检测评估
    支持相对路径和参数化配置
    """
    parser = argparse.ArgumentParser(
        description="3D目标检测评估器（NuScenes风格）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例用法：
  # 评估单个预测文件夹
  python this_script.py --data_root ./data/heightfusion/point1 \\
      --gt_folder training/label_2 --calib_folder training/calib \\
      --pred_folder outputs/heightfusion++_lidar/point1/eval_epoch=2499/data

  # 评估多个预测文件夹
  python this_script.py --data_root ./data/heightfusion/point1 \\
      --pred_folders outputs/folder1 outputs/folder2 outputs/folder3
        """
    )
    
    parser.add_argument('--data_root', type=str, default='./data/heightfusion/point1', help='数据集根目录（相对于项目根目录）')
    parser.add_argument('--gt_folder', type=str, default='training/label_2', help='GT标签文件夹（相对于data_root或绝对路径）')
    parser.add_argument('--calib_folder', type=str, default='training/calib', help='标定文件夹（相对于data_root或绝对路径）')
    parser.add_argument('--pred_folder', type=str, default=None, help='单个预测文件夹路径')
    parser.add_argument('--pred_folders', type=str, nargs='+', default=None, help='多个预测文件夹路径列表')
    parser.add_argument('--img_width', type=int, default=2560, help='图像宽度（像素）')
    parser.add_argument('--img_height', type=int, default=1440, help='图像高度（像素）')
    parser.add_argument('--lidar_range', type=float, default=105.0, help='点云有效范围（米）')
    parser.add_argument('--classes', type=str, nargs='+', default=['Car', 'Cyclist', 'Pedestrian'], help='要评估的类别列表')
    
    args = parser.parse_args()
    
    # 构建完整的路径
    def resolve_path(base, relative_path):
        """如果relative_path是绝对路径则返回原路径，否则相对于base"""
        if os.path.isabs(relative_path):
            return Path(relative_path)
        return Path(base) / relative_path
    
    data_root = Path(args.data_root)
    gt_folder = resolve_path(data_root, args.gt_folder)
    calib_folder = resolve_path(data_root, args.calib_folder)
    
    print(f"数据配置:")
    print(f"  数据根目录: {data_root}")
    print(f"  GT文件夹: {gt_folder}")
    print(f"  标定文件夹: {calib_folder}")
    
    # 验证路径存在性
    if not gt_folder.exists():
        print(f"错误：GT文件夹不存在: {gt_folder}")
        return
    
    if calib_folder and not calib_folder.exists():
        print(f"警告：标定文件夹不存在: {calib_folder}")
        calib_folder = None
    
    # 处理预测文件夹
    if args.pred_folder:
        pred_folders = [args.pred_folder]
        use_single_folder = True
    elif args.pred_folders:
        pred_folders = args.pred_folders
        use_single_folder = False
    else:
        print("错误：必须提供 --pred_folder 或 --pred_folders")
        parser.print_help()
        return
    
    # 初始化评估器
    evaluator = CustomKITTIEvaluator(
        gt_folder=str(gt_folder),
        pred_folders=pred_folders if not use_single_folder else None,
        pred_folder=pred_folders[0] if use_single_folder else None,
        calib_folder=str(calib_folder) if calib_folder else None,
        img_width=args.img_width,
        img_height=args.img_height,
        lidar_range=args.lidar_range,
        classes=args.classes
    )
    
    # 执行评估
    if use_single_folder:
        print("\n开始单个文件夹评估...")
        results, _ = evaluator.evaluate()
        
        if results:
            evaluator.visualize_results(results, output_dir="evaluation_results", folder_name=pred_folders[0])
            
            print("\n" + "="*80)
            print("评估结果汇总:")
            print("="*80)
            print(f"| mAP    | {results['mAP']:.4f} |")
            print(f"| NDS    | {results['NDS']:.4f} |")
            print(f"| mATE   | {results['nuscenes_metrics']['mATE']:.4f} m |")
            print(f"| mASE   | {results['nuscenes_metrics']['mASE']:.4f} |")
            print(f"| mAOE   | {results['nuscenes_metrics']['mAOE']:.4f} rad |")
    else:
        print(f"\n开始多文件夹评估（{len(pred_folders)} 个文件夹）...")
        best_folder, best_mAP, best_result, all_results = evaluator.evaluate_multiple_folders()
        
        if best_result:
            print(f"\n找到最佳预测文件夹: {best_folder}")
            print(f"最佳 mAP: {best_mAP:.4f}")
            
            evaluator.visualize_results(best_result, output_dir="evaluation_results", folder_name=best_folder)
            
            os.makedirs("evaluation_results", exist_ok=True)
            write_filename = best_folder.split('/')[1] if '/' in best_folder else "best"
            
            with open(f"evaluation_results/all_results_summary_{write_filename}.txt", 'w') as f:
                f.write("="*80 + "\n")
                f.write("所有预测文件夹评估结果汇总\n")
                f.write("="*80 + "\n\n")
                f.write(f"最佳预测文件夹: {best_folder}\n")
                f.write(f"最佳 mAP: {best_mAP:.4f}\n")
                f.write(f"最佳 NDS: {best_result['NDS']:.4f}\n\n")
                
                sorted_items = sorted(all_results.items(), key=lambda x: x[1]['mAP'], reverse=True)
                for i, (folder, data) in enumerate(sorted_items):
                    f.write(f"{i+1}. {folder}\n")
                    f.write(f"   mAP: {data['mAP']:.4f}, NDS: {data['NDS']:.4f}\n")
            
            print(f"\n结果已保存到 evaluation_results/")



if __name__ == "__main__":
    main()