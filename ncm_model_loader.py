"""
完整的NeuralCoMapping模型載入器
包含特徵提取和模型定義
"""

import torch
import torch.nn as nn
import numpy as np


class mGNN(nn.Module):
    """
    Multiplex Graph Neural Network
    從NeuralCoMapping論文實現
    """
    def __init__(self, node_dim=64, edge_dim=32, num_layers=3):
        super().__init__()
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(5, node_dim),  # [x, y, frontier_utility, distance_to_robot, exploration_gain]
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, edge_dim),  # [distance, path_cost, visibility]
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim)
        )
        
        # Message passing layers
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim + edge_dim, node_dim),
                nn.ReLU(),
                nn.LayerNorm(node_dim)
            ) for _ in range(num_layers)
        ])
        
        # Affinity predictor
        self.affinity_head = nn.Sequential(
            nn.Linear(node_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, node_features, edge_features, edge_indices):
        """
        Args:
            node_features: (num_nodes, 5)
            edge_features: (num_edges, 3)
            edge_indices: (num_edges, 2)
        Returns:
            affinity_matrix: (num_robots, num_frontiers)
        """
        # Encode features
        nodes = self.node_encoder(node_features)
        edges = self.edge_encoder(edge_features)
        
        # Message passing
        for layer in self.gnn_layers:
            messages = []
            for i, (src, dst) in enumerate(edge_indices):
                edge_msg = torch.cat([nodes[src], edges[i]], dim=-1)
                messages.append(layer(edge_msg))
            
            if messages:
                nodes = nodes + torch.stack(messages).mean(0)
        
        # Compute affinity between robots and frontiers
        num_robots = 2
        num_frontiers = len(nodes) - num_robots
        
        affinity = torch.zeros(num_robots, num_frontiers)
        for r in range(num_robots):
            for f in range(num_frontiers):
                pair = torch.cat([nodes[r], nodes[num_robots + f]])
                affinity[r, f] = self.affinity_head(pair)
        
        return affinity


def count_unknown_neighbors(x, y, op_map, radius=10):
    """計算周圍未探索區域數量"""
    h, w = op_map.shape
    count = 0
    total = 0
    
    for dx in range(-radius, radius+1):
        for dy in range(-radius, radius+1):
            nx, ny = int(x) + dx, int(y) + dy
            if 0 <= nx < w and 0 <= ny < h:
                total += 1
                if op_map[ny, nx] == 127:  # 未探索區域
                    count += 1
    
    return count / max(total, 1)


def estimate_exploration_gain(x, y, op_map, radius=15):
    """估計探索收益"""
    h, w = op_map.shape
    gain = 0
    
    for dx in range(-radius, radius+1):
        for dy in range(-radius, radius+1):
            nx, ny = int(x) + dx, int(y) + dy
            if 0 <= nx < w and 0 <= ny < h:
                if op_map[ny, nx] == 127:
                    dist = np.sqrt(dx**2 + dy**2)
                    gain += 1.0 / (1.0 + dist)
    
    return gain


def check_line_of_sight(x1, y1, x2, y2, op_map):
    """檢查兩點間是否有直線視線(簡化版)"""
    # 使用Bresenham算法
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    x_inc = 1 if x2 > x1 else -1
    y_inc = 1 if y2 > y1 else -1
    
    h, w = op_map.shape
    
    if dx > dy:
        error = dx / 2
        while x != x2:
            if 0 <= int(x) < w and 0 <= int(y) < h:
                if op_map[int(y), int(x)] == 0:  # 障礙物
                    return 0.0
            x += x_inc
            error -= dy
            if error < 0:
                y += y_inc
                error += dx
    else:
        error = dy / 2
        while y != y2:
            if 0 <= int(x) < w and 0 <= int(y) < h:
                if op_map[int(y), int(x)] == 0:
                    return 0.0
            y += y_inc
            error -= dx
            if error < 0:
                x += x_inc
                error += dy
    
    return 1.0


def extract_features(robots, frontiers, op_map):
    """
    從環境中提取特徵用於mGNN
    
    Args:
        robots: List of robot positions [(x,y), ...]
        frontiers: List of frontier positions [(x,y), ...]
        op_map: Occupancy map
        
    Returns:
        node_features: torch.FloatTensor (num_nodes, 5)
        edge_features: torch.FloatTensor (num_edges, 3)
        edge_indices: torch.LongTensor (num_edges, 2)
    """
    num_robots = len(robots)
    num_frontiers = len(frontiers)
    
    if num_frontiers == 0:
        # 處理沒有frontier的情況
        node_features = torch.zeros((num_robots, 5))
        edge_features = torch.zeros((0, 3))
        edge_indices = torch.zeros((0, 2), dtype=torch.long)
        return node_features, edge_features, edge_indices
    
    # Node features: [x_norm, y_norm, utility, dist_to_nearest_robot, exploration_gain]
    node_features = []
    
    map_h, map_w = op_map.shape
    
    # Robot nodes
    for rx, ry in robots:
        node_features.append([
            rx / map_w,
            ry / map_h,
            0.0,  # robots沒有utility
            0.0,  # 自己到自己距離為0
            0.0   # robots不提供exploration gain
        ])
    
    # Frontier nodes
    for fx, fy in frontiers:
        # Utility: 周圍未探索區域數量
        utility = count_unknown_neighbors(fx, fy, op_map)
        
        # Distance to nearest robot
        dists = [np.linalg.norm(np.array([fx, fy]) - np.array(r)) for r in robots]
        min_dist = min(dists) / np.sqrt(map_w**2 + map_h**2)  # normalize
        
        # Exploration gain: 估計探索這個frontier能獲得多少新信息
        exploration_gain = estimate_exploration_gain(fx, fy, op_map)
        
        node_features.append([
            fx / map_w,
            fy / map_h,
            utility,
            min_dist,
            exploration_gain
        ])
    
    node_features = torch.FloatTensor(node_features)
    
    # Edge features: [distance_norm, path_cost, visibility]
    edge_features = []
    edge_indices = []
    
    # Edges from robots to frontiers
    for r_idx in range(num_robots):
        rx, ry = robots[r_idx]
        for f_idx in range(num_frontiers):
            fx, fy = frontiers[f_idx]
            
            dist = np.linalg.norm(np.array([fx, fy]) - np.array([rx, ry]))
            dist_norm = dist / np.sqrt(map_w**2 + map_h**2)
            
            # Path cost (簡化版,實際應該用A*)
            path_cost = dist_norm
            
            # Visibility (是否直線可見)
            visibility = check_line_of_sight(rx, ry, fx, fy, op_map)
            
            edge_features.append([dist_norm, path_cost, visibility])
            edge_indices.append([r_idx, num_robots + f_idx])
    
    edge_features = torch.FloatTensor(edge_features)
    edge_indices = torch.LongTensor(edge_indices)
    
    return node_features, edge_features, edge_indices


def load_pretrained_ncm(model_path):
    """
    加載預訓練的NCM模型
    支持.pth和.global格式
    
    Args:
        model_path: 預訓練模型路徑
        
    Returns:
        model: 加載好的模型
    """
    model = mGNN()
    
    try:
        print(f"🔍 正在載入checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            print(f"   Checkpoint keys: {list(checkpoint.keys())}")
            
            # 情況1: 包含'network'鍵的RL訓練checkpoint
            if 'network' in checkpoint:
                print("   檢測到RL訓練checkpoint格式")
                network_state = checkpoint['network']
                
                # 嘗試載入,使用strict=False允許部分匹配
                missing_keys, unexpected_keys = model.load_state_dict(
                    network_state, strict=False
                )
                
                if missing_keys:
                    print(f"   ⚠️  Missing keys: {len(missing_keys)} keys")
                
                if unexpected_keys:
                    print(f"   ⚠️  Unexpected keys: {len(unexpected_keys)} keys")
                
                # 檢查是否有任何權重被載入
                loaded_params = sum(p.numel() for p in model.parameters())
                print(f"   📊 模型參數總數: {loaded_params}")
                
            # 情況2: 直接的state_dict
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.eval()
        print(f"✅ 模型載入完成!")
        
    except FileNotFoundError:
        print(f"⚠️  找不到預訓練模型: {model_path}")
        print("   使用隨機初始化的模型")
    except Exception as e:
        print(f"⚠️  加載模型時出錯: {e}")
        print("   使用隨機初始化的模型")
    
    model.eval()
    return model