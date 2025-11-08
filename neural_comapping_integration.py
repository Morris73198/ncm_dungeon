"""
Multi-Robot Environment with NeuralCoMapping Integration
將NeuralCoMapping的global planner替換原本的DQN frontier selection
[新增] 優先選擇 sensor_range * 2 以內的 frontier
[新增] 確保兩台機器人選點不會太近
[修復] 索引錯誤
"""

import numpy as np
from neural_comapping_adapter import NeuralCoMappingPlanner
from two_robot_dueling_dqn_attention.config import ROBOT_CONFIG


class RobotWithNeuralCoMapping:
    def __init__(self, original_robot, neural_planner):
        self.robot = original_robot
        self.planner = neural_planner
        self.other_robot_wrapper = None
        
        # [新增] 儲存 NCM 分配的長期目標
        self.current_global_goal = None
        
        # [新增] 全局規劃的頻率
        self.replanning_frequency = 10  # 每 10 步重新指派一次
        
        # [新增] 最小目標間距離（避免兩機器人選點太近）
        self.min_target_separation = self.robot.sensor_range * 1.5  # 預設 75
    
    def set_other_robot(self, other_wrapper):
        """設定另一個機器人的wrapper"""
        self.other_robot_wrapper = other_wrapper
    
    def filter_frontiers_by_range(self, frontiers, max_range=None):
        """
        [新增] 過濾出範圍內的 frontiers
        
        Args:
            frontiers: numpy array (N, 2) - 所有 frontier 點
            max_range: 最大距離（預設為 sensor_range * 2）
            
        Returns:
            in_range_frontiers: 範圍內的 frontiers
            out_range_frontiers: 範圍外的 frontiers
        """
        if max_range is None:
            max_range = self.robot.sensor_range * 2
        
        # 計算每個 frontier 到機器人的距離
        distances = np.linalg.norm(frontiers - self.robot.robot_position, axis=1)
        
        # 分類為範圍內和範圍外
        in_range_mask = distances <= max_range
        in_range_frontiers = frontiers[in_range_mask]
        out_range_frontiers = frontiers[~in_range_mask]
        
        return in_range_frontiers, out_range_frontiers
    
    def adjust_assignments_for_separation(self, assignments, frontiers_array, robots):
        """
        [新增] 調整分配結果，確保兩個機器人的目標點不會太近
        [修復] 正確處理 assignments 中已經是座標而非索引的情況
        
        Args:
            assignments: {robot_idx: (x, y)} - NCM 的原始分配（已經是座標）
            frontiers_array: numpy array (N, 2) - 所有 frontier 座標
            robots: list of tuples - 兩個機器人的位置
            
        Returns:
            adjusted_assignments: 調整後的分配
        """
        # 如果只有一個機器人被分配，或沒有分配，直接返回
        if len(assignments) < 2:
            return assignments
        
        # 獲取兩個機器人的分配（注意：assignments 的值已經是座標 tuple，不是索引）
        if 0 not in assignments or 1 not in assignments:
            return assignments
        
        target_0 = np.array(assignments[0])
        target_1 = np.array(assignments[1])
        
        # 計算兩個目標點之間的距離
        target_distance = np.linalg.norm(target_0 - target_1)
        
        robot_name = "Robot1" if self.robot.is_primary else "Robot2"
        
        # 如果距離太近，需要調整
        if target_distance < self.min_target_separation:
            print(f"[NCM SEPARATION] {robot_name}: 目標距離 {target_distance:.1f} < {self.min_target_separation:.1f}，需要調整")
            
            # 找出應該被調整的機器人（優先調整離自己遠的那個）
            dist_0_to_target_0 = np.linalg.norm(np.array(robots[0]) - target_0)
            dist_1_to_target_1 = np.linalg.norm(np.array(robots[1]) - target_1)
            
            # 決定調整哪個機器人的目標
            # 調整離目標較遠的機器人（因為對它影響較小）
            robot_to_adjust = 0 if dist_0_to_target_0 > dist_1_to_target_1 else 1
            other_robot = 1 - robot_to_adjust
            other_target = target_0 if other_robot == 0 else target_1
            
            # 尋找替代目標（距離當前機器人近，但距離另一機器人目標遠）
            robot_pos = np.array(robots[robot_to_adjust])
            
            # 計算所有 frontier 到當前機器人的距離
            dists_to_robot = np.linalg.norm(frontiers_array - robot_pos, axis=1)
            # 計算所有 frontier 到另一機器人目標的距離
            dists_to_other_target = np.linalg.norm(frontiers_array - other_target, axis=1)
            
            # 過濾掉距離另一機器人目標太近的點
            valid_mask = dists_to_other_target >= self.min_target_separation
            
            if np.any(valid_mask):
                # 在有效點中選擇距離當前機器人最近的
                valid_indices = np.where(valid_mask)[0]
                valid_dists = dists_to_robot[valid_indices]
                best_idx = valid_indices[np.argmin(valid_dists)]
                
                # 更新分配（使用座標）
                new_target = tuple(frontiers_array[best_idx])
                assignments[robot_to_adjust] = new_target
                print(f"[NCM SEPARATION] 調整 Robot{robot_to_adjust+1} 的目標 (新距離: {dists_to_robot[best_idx]:.1f})")
            else:
                print(f"[NCM SEPARATION] 警告：找不到符合距離要求的替代目標，保持原分配")
        else:
            print(f"[NCM SEPARATION] {robot_name}: 目標距離 {target_distance:.1f} ✓ (≥ {self.min_target_separation:.1f})")
        
        return assignments
    
    def step_with_neural_planner(self):
        """
        [全新修正] 實現真正的 NCM 分層規劃邏輯
        [新增] 優先選擇 sensor_range * 2 以內的 frontier
        [新增] 確保兩台機器人選點不會太近
        """
        
        # 檢查是否需要執行「全局規劃」(NCM 分配)
        # 條件：計數器到期 OR 機器人沒有長期目標
        if (self.robot.steps % self.replanning_frequency == 0) or (self.current_global_goal is None):
            
            # --- 1. 全局規劃器 (NCM) ---
            all_frontiers = self.robot.get_frontiers()
            
            if len(all_frontiers) == 0:
                return None, self.robot.check_done()

            # [新增] 優先選擇範圍內的 frontiers
            max_range = self.robot.sensor_range * 2  # 100
            in_range_frontiers, out_range_frontiers = self.filter_frontiers_by_range(
                all_frontiers, max_range
            )
            
            robot_name = "Robot1" if self.robot.is_primary else "Robot2"
            
            # 優先使用範圍內的 frontiers
            if len(in_range_frontiers) > 0:
                frontiers = in_range_frontiers
                print(f"[NCM RANGE] {robot_name}: 使用 {len(in_range_frontiers)} 個範圍內的 frontiers (距離 ≤ {max_range})")
            else:
                frontiers = all_frontiers
                print(f"[NCM RANGE] {robot_name}: 範圍內無 frontiers，使用全部 {len(all_frontiers)} 個")

            robots = [
                tuple(self.robot.robot_position),
                tuple(self.robot.other_robot_position)
            ]

            # NCM 分配（返回的是 {robot_idx: (x, y)} 形式）
            assignments = self.planner.select_frontiers(
                robots, 
                frontiers,
                self.robot.op_map
            )
            
            # [新增] 調整分配，確保兩個機器人的目標不會太近
            # 注意：assignments 已經是 {robot_idx: (x, y)} 形式，不需要再轉換
            adjusted_assignments = self.adjust_assignments_for_separation(
                assignments, 
                frontiers,  # 直接傳遞 numpy array
                robots
            )
            
            robot_idx = 0 if self.robot.is_primary else 1
            
            if robot_idx not in adjusted_assignments:
                print(f"[NCM DEBUG] {robot_name}: NCM 未分配, 啟動 [後援-最近點]")
                
                # 後援策略也要考慮與另一機器人的距離
                dists = np.linalg.norm(frontiers - self.robot.robot_position, axis=1)
                
                # 如果另一個機器人有目標，避開它
                if self.other_robot_wrapper and self.other_robot_wrapper.current_global_goal is not None:
                    other_goal = self.other_robot_wrapper.current_global_goal
                    dists_to_other = np.linalg.norm(frontiers - other_goal, axis=1)
                    
                    # 過濾掉離另一機器人目標太近的點
                    valid_mask = dists_to_other >= self.min_target_separation
                    if np.any(valid_mask):
                        valid_indices = np.where(valid_mask)[0]
                        new_target = frontiers[valid_indices[np.argmin(dists[valid_indices])]]
                        print(f"[NCM DEBUG] {robot_name}: 後援選點時避開另一機器人")
                    else:
                        new_target = frontiers[np.argmin(dists)]
                else:
                    new_target = frontiers[np.argmin(dists)]
            else:
                assigned_target = np.array(adjusted_assignments[robot_idx])
                dist_to_target = np.linalg.norm(assigned_target - self.robot.robot_position)
                print(f"[NCM DEBUG] {robot_name}: NCM 分配新目標 (距離: {dist_to_target:.1f})")
                new_target = assigned_target
            
            # [關鍵] 更新長期目標
            self.current_global_goal = new_target
            # 確保 robot 物件也更新它，這樣 move_to_frontier 才能正確規劃
            self.robot.current_target_frontier = self.current_global_goal

        # --- 2. 局部規劃器 (每一步都執行) ---
        
        if self.current_global_goal is None:
            # 即使 NCM 失敗了，也沒找到後援點 (例如地圖是空的)
            return None, self.robot.check_done()

        # [關鍵] 無論如何，都朝著「儲存的」長期目標移動一步
        # move_to_frontier 內部會自己處理 A* 尋路
        observation, reward, task_done = self.robot.move_to_frontier(self.current_global_goal)
        
        # 如果 move_to_frontier 說 "done" (表示它抵達了)，
        # 我們就把長期目標設為 None，強制 NCM 在下一步重新規劃
        if task_done:
            self.current_global_goal = None

        # 檢查 Episode 是否結束 (探索率是否達標)
        episode_done = self.robot.check_done()
        
        # (地圖同步)
        if hasattr(self.robot, 'shared_env') and self.robot.shared_env is not None:
            self.robot.shared_env.op_map = self.robot.op_map
        elif hasattr(self.robot, 'other_robot') and self.robot.other_robot is not None:
            self.robot.other_robot.op_map = self.robot.op_map

        # (更新步數和繪圖)
        self.robot.steps += 1
        if self.robot.plot and self.robot.steps % 5 == 0:
            self.robot.plot_env()
            import matplotlib.pyplot as plt
            plt.pause(0.01)

        return observation, episode_done
    
    def _execute_movement_step(self):
        """
        執行一步移動 (使用原有的移動邏輯)
        !! 已修正為逐步移動 !!
        """
        if not self.robot.is_moving_to_target or self.robot.current_path is None:
            self.robot.is_moving_to_target = False
            return None, False
        
        if self.robot.current_path_index >= self.robot.current_path.shape[1]:
            # 路徑上的所有點都已訪問
            self.robot.is_moving_to_target = False
            
            # 檢查是否真的到達最終目標
            dist_to_target = np.linalg.norm(self.robot.robot_position - self.robot.current_target_frontier)
            
            # 從 config.py 獲取閾值 (如果不存在則默認為 10)
            target_reach_threshold = ROBOT_CONFIG.get('target_reach_threshold', 10)

            if dist_to_target < target_reach_threshold:
                 # 真的到了，強制掃描一次以清除這個frontier
                self.robot.op_map = self.robot.inverse_sensor(
                    self.robot.robot_position, self.robot.sensor_range,
                    self.robot.op_map, self.robot.global_map
                )
            
            return None, False # 移動結束

        # --- (從 multi_robot.py 複製並修改的 "正確" 邏輯) ---
        
        # 1. 獲取路徑上的 *下一個* 檢查點
        next_point = self.robot.current_path[:, self.robot.current_path_index]
        
        # 2. 計算朝向下一個檢查點的 *移動向量*
        move_vector = next_point - self.robot.robot_position
        dist = np.linalg.norm(move_vector)
        
        # 3. 從 config.py 獲取步長 (如果不存在則默認為 2)
        movement_step = ROBOT_CONFIG.get('movement_step', 2) 
        
        # 4. 確保最小移動 (如果離下一個檢查點太近，直接跳到下下個)
        MIN_MOVEMENT = 1.0
        if dist < MIN_MOVEMENT:
            self.robot.current_path_index += 1
            # 只是推進了索引，還沒移動，所以回傳 False (未完成)
            return self.robot.get_observation(), False 

        # 5. 限制這一步的長度 (關鍵！)
        if dist > movement_step:
            move_vector = move_vector * (movement_step / dist)
        
        # 6. 執行這 "一小步" 移動
        old_position = self.robot.robot_position.copy()
        new_position = self.robot.robot_position + move_vector
        self.robot.robot_position = np.round(new_position).astype(np.int64)
        
        # 7. 邊界檢查
        self.robot.robot_position[0] = np.clip(self.robot.robot_position[0], 0, self.robot.map_size[1]-1)
        self.robot.robot_position[1] = np.clip(self.robot.robot_position[1], 0, self.robot.map_size[0]-1)

        # 8. 碰撞檢查 (使用觀測地圖 op_map)
        #    (注意：原版 multi_robot.py 使用 global_map 檢查，這裡用 op_map 可能更合理)
        if self.robot.op_map[self.robot.robot_position[1], self.robot.robot_position[0]] == 1:
            self.robot.robot_position = old_position # 撤銷移動
            self.robot.is_moving_to_target = False # 路徑被擋，停止
            self.robot.current_path = None # 清除路徑，下次重新規劃
            return self.robot.get_observation(), False # 沒完成，但路徑失敗

        # 9. 更新地圖 (!! 修正：同時呼叫 robot_model 和 inverse_sensor !!)
        self.robot.op_map = self.robot.robot_model(
            self.robot.robot_position, self.robot.robot_size,
            self.robot.t, self.robot.op_map
        )
        self.robot.op_map = self.robot.inverse_sensor(
            self.robot.robot_position, self.robot.sensor_range,
            self.robot.op_map, self.robot.global_map
        )
        
        # 10. (地圖同步)
        if hasattr(self.robot, 'shared_env') and self.robot.shared_env is not None:
            self.robot.shared_env.op_map = self.robot.op_map
        elif hasattr(self.robot, 'other_robot') and self.robot.other_robot is not None:
            self.robot.other_robot.op_map = self.robot.op_map
        
        # 11. (記錄軌跡)
        self.robot.xPoint = np.append(self.robot.xPoint, self.robot.robot_position[0])
        self.robot.yPoint = np.append(self.robot.yPoint, self.robot.robot_position[1])
        
        # 12. 只有當離下一個檢查點足夠近時，才推進 path_index
        target_reach_threshold = ROBOT_CONFIG.get('target_reach_threshold', 10)
        if dist < target_reach_threshold or dist < movement_step:
             self.robot.current_path_index += 1

        # 13. (更新步數和繪圖)
        self.robot.steps += 1
        done = self.robot.check_done()
        observation = self.robot.get_observation()
        
        if self.robot.plot and self.robot.steps % 5 == 0:
            self.robot.plot_env()
            import matplotlib.pyplot as plt
            plt.pause(0.01)
        
        return observation, done


def create_neural_comapping_robots(index_map=0, use_neural=False, model_path=None):
    """
    創建使用NeuralCoMapping的雙機器人系統
    """
    from two_robot_dueling_dqn_attention.environment.multi_robot import Robot
    
    robot1, robot2 = Robot.create_shared_robots(
        index_map=index_map,
        train=False,
        plot=True
    )
    
    planner = NeuralCoMappingPlanner(
        use_neural=use_neural,
        model_path=model_path
    )
    
    robot1_wrapper = RobotWithNeuralCoMapping(robot1, planner)
    robot2_wrapper = RobotWithNeuralCoMapping(robot2, planner)
    
    robot1_wrapper.set_other_robot(robot2_wrapper)
    robot2_wrapper.set_other_robot(robot1_wrapper)
    
    return robot1_wrapper, robot2_wrapper


def run_episode_with_neural_comapping(robot1_wrapper, robot2_wrapper, max_steps=1000):
    """使用NeuralCoMapping運行一個episode"""
    robot1_wrapper.robot.reset()
    robot2_wrapper.robot.reset()
    
    done1 = done2 = False
    steps = 0
    
    print(f"  開始探索...", end='', flush=True)
    
    while steps < max_steps and not (done1 and done2):
        if not done1:
            _, done1 = robot1_wrapper.step_with_neural_planner()
        
        if not done2:
            _, done2 = robot2_wrapper.step_with_neural_planner()
        
        steps += 1
        
        if steps % 100 == 0:
            exploration_ratio = np.sum(robot1_wrapper.robot.op_map == 255) / \
                               np.sum(robot1_wrapper.robot.global_map == 255)
            print(f"\r  步數: {steps}, 探索率: {exploration_ratio:.1%}", end='', flush=True)
        
        if robot1_wrapper.robot.plot and steps % 50 == 0:
            import matplotlib.pyplot as plt
            robot1_wrapper.robot.plot_env()
            plt.pause(0.01)
    
    print()
    
    exploration_ratio = np.sum(robot1_wrapper.robot.op_map == 255) / \
                       np.sum(robot1_wrapper.robot.global_map == 255)
    
    return steps, exploration_ratio