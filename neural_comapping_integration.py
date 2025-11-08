"""
Multi-Robot Environment with NeuralCoMapping Integration
將NeuralCoMapping的global planner替換原本的DQN frontier selection
"""

import numpy as np
from neural_comapping_adapter import NeuralCoMappingPlanner


class RobotWithNeuralCoMapping:
    """
    將NeuralCoMapping整合到現有Robot class的wrapper
    只替換frontier selection部分,保留A*和其他模組
    """
    
    def __init__(self, original_robot, neural_planner):
        """
        Args:
            original_robot: 你原本的Robot實例
            neural_planner: NeuralCoMappingPlanner實例
        """
        self.robot = original_robot
        self.planner = neural_planner
        
        # 記錄配對的另一個機器人(用於雙機器人系統)
        self.other_robot_wrapper = None
    
    def set_other_robot(self, other_wrapper):
        """設定另一個機器人的wrapper"""
        self.other_robot_wrapper = other_wrapper
    
    def step_with_neural_planner(self):
        """
        使用NeuralCoMapping選擇frontier,然後用A*規劃路徑
        替代原本的DQN action selection
        """
        # 如果正在執行路徑,繼續執行
        if (self.robot.is_moving_to_target and 
            self.robot.current_path is not None and 
            self.robot.current_path_index < self.robot.current_path.shape[1]):
            return self._execute_movement_step()
        
        # 路徑執行完畢或沒有路徑,重新規劃
        # 1. 獲取frontiers
        frontiers = self.robot.get_frontiers()
        
        if len(frontiers) == 0:
            return None, True  # 沒有frontier,探索完成
        
        # 2. 準備兩個機器人的位置
        robots = [
            tuple(self.robot.robot_position),
            tuple(self.robot.other_robot_position)
        ]
        
        # 3. 使用NeuralCoMapping選擇frontier
        assignments = self.planner.select_frontiers(
            robots, 
            frontiers, 
            self.robot.op_map
        )
        
        # 4. 獲取此機器人的目標frontier
        robot_idx = 0 if self.robot.is_primary else 1
        
        if robot_idx not in assignments:
            # 如果沒有分配給此機器人,選距離最近的
            dists = np.linalg.norm(frontiers - self.robot.robot_position, axis=1)
            target_frontier = frontiers[np.argmin(dists)]
        else:
            target_frontier = np.array(assignments[robot_idx])

        
        # 5. 使用A*規劃到目標frontier的路徑
        self.robot.current_target_frontier = target_frontier
        
        # 使用原有的A*路徑規劃 (使用astar_path而非astar)
        if hasattr(self.robot, 'astar_path'):
            # 如果有astar_path方法,使用它
            path = self.robot.astar_path(
                self.robot.op_map,
                self.robot.robot_position.astype(np.int32),
                target_frontier.astype(np.int32)
            )
        else:
            # 否則使用astar
            path = self.robot.astar(
                self.robot.op_map,
                self.robot.robot_position,
                target_frontier
            )
        
        if path is None:
            # 路徑規劃失敗,標記為不可移動
            print(f"Robot{robot_idx+1}: Path planning failed! (path is None)")
            self.robot.is_moving_to_target = False
            return None, False
        
        # 檢查path長度
        path_length = len(path) if isinstance(path, list) else (path.shape[1] if path.ndim == 2 else path.shape[0])
        
        if path_length <= 1:
            # 路徑太短(只有起點),認為已經到達frontier
            # 強制執行感測器掃描以更新地圖並移除這個frontier - 使用in-place更新
            updated_map = self.robot.inverse_sensor(
                self.robot.robot_position,
                self.robot.sensor_range,
                self.robot.op_map,
                self.robot.global_map
            )
            
            # In-place更新以保持共享引用
            np.copyto(self.robot.op_map, updated_map)
            
            self.robot.is_moving_to_target = False
            
            # 重新獲取frontiers看是否有新的
            new_frontiers = self.robot.get_frontiers()
            if len(new_frontiers) == 0:
                return None, True  # 沒有frontier了,完成探索
            
            # 立即規劃下一個目標
            return None, False
        
        print(f"Robot{robot_idx+1}: Path found with {path_length} points, moving...")
        
        # 6. 設置路徑並開始移動
        # 檢查path的格式並轉換
        if isinstance(path, np.ndarray):
            if path.ndim == 1:
                # 1D array
                self.robot.current_path = path.reshape(2, -1)
            elif path.shape[0] == 2:
                # Already in correct format (2, N)
                self.robot.current_path = path
            else:
                # (N, 2) format, need to transpose
                self.robot.current_path = path.T
        else:
            # List of tuples
            self.robot.current_path = np.array(path).T
            
        self.robot.current_path_index = 0
        self.robot.is_moving_to_target = True
        
        # 執行第一步移動
        return self._execute_movement_step()
    
    def _execute_movement_step(self):
        """
        執行一步移動(使用原有的移動邏輯)
        每次移動movement_step個單位
        """
        if not self.robot.is_moving_to_target or self.robot.current_path is None:
            return None, False
        
        # 檢查是否到達目標
        if self.robot.current_path_index >= self.robot.current_path.shape[1]:
            self.robot.is_moving_to_target = False
            return None, False
        
        # 獲取下一個目標點(跳躍movement_step個點)
        movement_step = getattr(self.robot, 'movement_step', 5)
        next_index = min(self.robot.current_path_index + movement_step, 
                        self.robot.current_path.shape[1] - 1)
        
        next_point = self.robot.current_path[:, next_index]
        
        # 移動機器人
        self.robot.robot_position = next_point.astype(np.int64)
        self.robot.current_path_index = next_index + 1
        
        # 更新地圖(使用原有的感測器模型) - 使用in-place更新
        updated_map = self.robot.robot_model(
            self.robot.robot_position,
            self.robot.robot_size,
            self.robot.t,
            self.robot.op_map
        )
        
        # In-place更新以保持共享引用
        np.copyto(self.robot.op_map, updated_map)
        
        # 記錄軌跡
        self.robot.xPoint = np.append(self.robot.xPoint, self.robot.robot_position[0])
        self.robot.yPoint = np.append(self.robot.yPoint, self.robot.robot_position[1])
        
        # 更新步數
        self.robot.steps += 1
        
        # 檢查是否完成探索
        done = self.robot.check_done()
        
        # 獲取觀測
        observation = self.robot.get_observation()
        
        # 可視化
        if self.robot.plot and self.robot.steps % 5 == 0:
            self.robot.plot_env()
            import matplotlib.pyplot as plt
            plt.pause(0.01)  # 短暫暫停以更新畫面
        
        return observation, done


def create_neural_comapping_robots(index_map=0, use_neural=False, model_path=None):
    """
    創建使用NeuralCoMapping的雙機器人系統
    
    Args:
        index_map: 地圖索引
        use_neural: 是否使用神經網路版本的matcher
        model_path: 神經網路模型路徑(如果use_neural=True)
        
    Returns:
        robot1_wrapper, robot2_wrapper: 兩個RobotWithNeuralCoMapping實例
    """
    # 導入你原本的Robot class
    # 這裡假設從你的環境導入
    from two_robot_dueling_dqn_attention.environment.multi_robot_no_unknown import Robot
    
    # 創建原始的機器人
    robot1, robot2 = Robot.create_shared_robots(
        index_map=index_map,
        train=False,
        plot=True
    )
    
    # 創建NeuralCoMapping planner
    planner = NeuralCoMappingPlanner(
        use_neural=use_neural,
        model_path=model_path
    )
    
    # 創建wrapper
    robot1_wrapper = RobotWithNeuralCoMapping(robot1, planner)
    robot2_wrapper = RobotWithNeuralCoMapping(robot2, planner)
    
    # 設定互相引用
    robot1_wrapper.set_other_robot(robot2_wrapper)
    robot2_wrapper.set_other_robot(robot1_wrapper)
    
    return robot1_wrapper, robot2_wrapper


def run_episode_with_neural_comapping(robot1_wrapper, robot2_wrapper, max_steps=1000):
    """
    使用NeuralCoMapping運行一個episode
    
    Args:
        robot1_wrapper: Robot1的wrapper
        robot2_wrapper: Robot2的wrapper
        max_steps: 最大步數
        
    Returns:
        total_steps: 總步數
        exploration_ratio: 探索率
    """
    robot1_wrapper.robot.reset()
    robot2_wrapper.robot.reset()
    
    done1 = done2 = False
    steps = 0
    
    print(f"  開始探索...", end='', flush=True)
    
    while steps < max_steps and not (done1 and done2):
        # Robot1執行一步
        if not done1:
            _, done1 = robot1_wrapper.step_with_neural_planner()
        
        # Robot2執行一步
        if not done2:
            _, done2 = robot2_wrapper.step_with_neural_planner()
        
        steps += 1
        
        # 每100步顯示進度
        if steps % 100 == 0:
            exploration_ratio = np.sum(robot1_wrapper.robot.op_map == 255) / \
                               np.sum(robot1_wrapper.robot.global_map == 255)
            print(f"\r  步數: {steps}, 探索率: {exploration_ratio:.1%}", end='', flush=True)
        
        # 可視化(減少頻率)
        if robot1_wrapper.robot.plot and steps % 50 == 0:
            import matplotlib.pyplot as plt
            robot1_wrapper.robot.plot_env()
            plt.pause(0.01)
    
    print()  # 換行
    
    # 計算最終探索率
    exploration_ratio = np.sum(robot1_wrapper.robot.op_map == 255) / \
                       np.sum(robot1_wrapper.robot.global_map == 255)
    
    return steps, exploration_ratio