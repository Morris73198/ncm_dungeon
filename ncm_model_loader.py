"""
加載NeuralCoMapping預訓練模型
需要從原始repo獲取模型權重
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
        # 嘗試直接加載
        if model_path.endswith('.global'):
            # .global文件是PyTorch全局權重
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # .global可能只是state_dict,也可能包含其他信息
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # 直接是state_dict
                model.load_state_dict(checkpoint)
        else:
            # .pth文件
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"✅ 成功加載預訓練模型: {model_path}")
    except FileNotFoundError:
        print(f"⚠️  找不到預訓練模型: {model_path}")
        print("   使用隨機初始化的模型 (性能會較差)")
    except Exception as e:
        print(f"⚠️  加載模型時出錯: {e}")
        print("   使用隨機初始化的模型")
    
    return model


def extract_features(robots, frontiers, op_map):
    """
    從環境中提取特徵用於mGNN
    
    Args:
        robots: List of robot positions [(x,y), ...]
        frontiers: Array of frontier positions (N, 2)
        op_map: Occupancy map
        
    Returns:
        node_features, edge_features, edge_indices
    """
    num_robots = len(robots)
    num_frontiers = len(frontiers)
    
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


def count_unknown_neighbors(x, y, op_map, radius=10):
    """計算周圍未知區域數量"""
    h, w = op_map.shape
    count = 0
    for dx in range(-radius, radius+1):
        for dy in range(-radius, radius+1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if op_map[ny, nx] == 127:  # 未知區域
                    count += 1
    return count / (radius * 2 + 1) ** 2


def estimate_exploration_gain(x, y, op_map, sensor_range=80):
    """估計探索收益"""
    # 簡化版: 計算sensor_range內的未知區域比例
    h, w = op_map.shape
    unknown = 0
    total = 0
    
    for dx in range(-sensor_range, sensor_range+1):
        for dy in range(-sensor_range, sensor_range+1):
            if dx*dx + dy*dy <= sensor_range*sensor_range:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    total += 1
                    if op_map[ny, nx] == 127:
                        unknown += 1
    
    return unknown / max(total, 1)


def check_line_of_sight(x1, y1, x2, y2, op_map):
    """檢查兩點間是否有直線可見(沒有障礙物)"""
    # Bresenham's line algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    x, y = x1, y1
    while True:
        if op_map[y, x] == 1:  # 障礙物
            return 0.0
        
        if x == x2 and y == y2:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    
    return 1.0


if __name__ == "__main__":
    # 測試
    model = load_pretrained_ncm("ncm_pretrained.pth")
    print(f"Model loaded: {model}")