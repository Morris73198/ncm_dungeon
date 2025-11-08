"""
NeuralCoMapping Adapter for Dungeon Simulator
簡化版的bipartite graph matching用於frontier selection
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn


class SimplifiedGraphMatcher:
    """
    簡化版的bipartite graph matching
    用於匹配機器人和frontiers
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        
    def compute_affinity_matrix(self, robots, frontiers, op_map):
        """
        計算機器人和frontier之間的affinity matrix
        
        Args:
            robots: list of robot positions [(x1,y1), (x2,y2), ...]
            frontiers: list of frontier positions [(x1,y1), (x2,y2), ...]
            op_map: 當前探索地圖
            
        Returns:
            affinity_matrix: shape (num_robots, num_frontiers)
        """
        num_robots = len(robots)
        num_frontiers = len(frontiers)
        
        if num_frontiers == 0:
            return np.zeros((num_robots, 0))
        
        affinity_matrix = np.zeros((num_robots, num_frontiers))
        
        for i, robot_pos in enumerate(robots):
            for j, frontier_pos in enumerate(frontiers):
                # 基於距離的affinity (距離越近分數越高)
                dist = np.linalg.norm(np.array(robot_pos) - np.array(frontier_pos))
                
                # 基於探索收益的affinity (frontier周圍未知區域越多分數越高)
                exploration_gain = self._compute_exploration_gain(frontier_pos, op_map)
                
                # 組合分數 (可調整權重)
                affinity_matrix[i, j] = exploration_gain / (dist + 1.0)
        
        return affinity_matrix
    
    def _compute_exploration_gain(self, frontier_pos, op_map, radius=10):
        """計算frontier周圍的探索收益"""
        x, y = int(frontier_pos[0]), int(frontier_pos[1])
        h, w = op_map.shape
        
        unknown_count = 0
        total_count = 0
        
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    total_count += 1
                    if op_map[ny, nx] == 127:  # 未知區域
                        unknown_count += 1
        
        return unknown_count / (total_count + 1e-6)
    
    def match(self, affinity_matrix):
        """
        使用Hungarian algorithm進行最優匹配
        
        Args:
            affinity_matrix: shape (num_robots, num_frontiers)
            
        Returns:
            matches: dict {robot_idx: frontier_idx}
        """
        if affinity_matrix.shape[1] == 0:
            return {}
        
        # Hungarian algorithm需要cost matrix (最小化)
        # 所以我們取負的affinity (最大化affinity = 最小化negative affinity)
        cost_matrix = -affinity_matrix
        
        robot_indices, frontier_indices = linear_sum_assignment(cost_matrix)
        
        matches = {}
        for robot_idx, frontier_idx in zip(robot_indices, frontier_indices):
            matches[robot_idx] = frontier_idx
            
        return matches


class NeuralGraphMatcher(nn.Module):
    """
    神經網路版本的graph matching (可選,用於訓練)
    使用GNN學習更好的affinity matrix
    """
    
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        # Robot feature encoder
        self.robot_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # [x, y, explored_area, distance_to_nearest_frontier]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Frontier feature encoder
        self.frontier_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),  # [x, y, exploration_gain]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Affinity predictor
        self.affinity_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, robot_features, frontier_features):
        """
        Args:
            robot_features: shape (num_robots, 4)
            frontier_features: shape (num_frontiers, 3)
            
        Returns:
            affinity_matrix: shape (num_robots, num_frontiers)
        """
        num_robots = robot_features.size(0)
        num_frontiers = frontier_features.size(0)
        
        # Encode features
        robot_embed = self.robot_encoder(robot_features)  # (num_robots, hidden_dim)
        frontier_embed = self.frontier_encoder(frontier_features)  # (num_frontiers, hidden_dim)
        
        # Compute pairwise affinity
        affinity_matrix = torch.zeros(num_robots, num_frontiers)
        
        for i in range(num_robots):
            for j in range(num_frontiers):
                # Concatenate robot and frontier embeddings
                pair_embed = torch.cat([robot_embed[i], frontier_embed[j]], dim=0)
                affinity_matrix[i, j] = self.affinity_predictor(pair_embed).squeeze()
        
        return affinity_matrix


class NeuralCoMappingPlanner:
    """
    完整的NeuralCoMapping planner
    結合簡化版matching和可選的neural matching
    """
    
    def __init__(self, use_neural=False, model_path=None):
        self.use_neural = use_neural
        self.simple_matcher = SimplifiedGraphMatcher()
        
        if use_neural:
            self.neural_matcher = NeuralGraphMatcher()
            if model_path:
                self.neural_matcher.load_state_dict(torch.load(model_path))
            self.neural_matcher.eval()
        else:
            self.neural_matcher = None
    
    def select_frontiers(self, robots, frontiers, op_map):
        """
        為每個機器人選擇最優的frontier
        
        Args:
            robots: list of robot positions [(x1,y1), (x2,y2)]
            frontiers: numpy array of frontier positions, shape (N, 2)
            op_map: 當前探索地圖
            
        Returns:
            assignments: dict {robot_idx: frontier_position}
        """
        if len(frontiers) == 0:
            return {}
        
        # Convert frontiers to list
        frontier_list = [tuple(f) for f in frontiers]
        
        if self.use_neural and self.neural_matcher is not None:
            # 使用神經網路計算affinity
            affinity_matrix = self._compute_neural_affinity(robots, frontier_list, op_map)
        else:
            # 使用簡化版計算affinity
            affinity_matrix = self.simple_matcher.compute_affinity_matrix(
                robots, frontier_list, op_map
            )
        
        # 進行匹配
        matches = self.simple_matcher.match(affinity_matrix)
        
        # 轉換為frontier positions
        assignments = {}
        for robot_idx, frontier_idx in matches.items():
            assignments[robot_idx] = frontier_list[frontier_idx]
        
        return assignments
    
    def _compute_neural_affinity(self, robots, frontiers, op_map):
        """使用神經網路計算affinity matrix"""
        # Extract features
        robot_features = self._extract_robot_features(robots, op_map, frontiers)
        frontier_features = self._extract_frontier_features(frontiers, op_map)
        
        # Convert to tensors
        robot_features = torch.FloatTensor(robot_features)
        frontier_features = torch.FloatTensor(frontier_features)
        
        # Compute affinity
        with torch.no_grad():
            affinity_matrix = self.neural_matcher(robot_features, frontier_features)
        
        return affinity_matrix.numpy()
    
    def _extract_robot_features(self, robots, op_map, frontiers):
        """提取機器人特徵"""
        features = []
        explored_ratio = np.sum(op_map == 255) / (op_map.shape[0] * op_map.shape[1])
        
        for robot_pos in robots:
            # 計算到最近frontier的距離
            if len(frontiers) > 0:
                dists = [np.linalg.norm(np.array(robot_pos) - np.array(f)) 
                        for f in frontiers]
                min_dist = min(dists)
            else:
                min_dist = 0
            
            features.append([
                robot_pos[0],
                robot_pos[1],
                explored_ratio,
                min_dist
            ])
        
        return np.array(features)
    
    def _extract_frontier_features(self, frontiers, op_map):
        """提取frontier特徵"""
        features = []
        
        for frontier_pos in frontiers:
            exploration_gain = self.simple_matcher._compute_exploration_gain(
                frontier_pos, op_map
            )
            
            features.append([
                frontier_pos[0],
                frontier_pos[1],
                exploration_gain
            ])
        
        return np.array(features)