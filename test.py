"""
測試NeuralCoMapping整合
比較DQN vs NeuralCoMapping的性能
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用互動式後端
import matplotlib.pyplot as plt
plt.ion()  # 開啟互動模式

from neural_comapping_integration import (
    create_neural_comapping_robots,
    run_episode_with_neural_comapping
)


def test_neural_comapping(num_episodes=10):
    """
    測試NeuralCoMapping在地牢環境中的表現
    
    Args:
        num_episodes: 測試的episode數量
    """
    print("=" * 60)
    print("測試NeuralCoMapping在地牢模擬器中的表現")
    print("=" * 60)
    
    # 創建使用NeuralCoMapping的機器人
    print("\n創建使用NeuralCoMapping的機器人...")
    robot1, robot2 = create_neural_comapping_robots(
        index_map=0,
        use_neural=True  # 先用簡化版測試
    )
    
    results = {
        'steps': [],
        'exploration_ratios': []
    }
    
    print(f"\n開始測試 {num_episodes} 個episodes...")
    
    for ep in range(num_episodes):
        print(f"\nEpisode {ep + 1}/{num_episodes}")
        
        # 運行episode
        steps, exploration_ratio = run_episode_with_neural_comapping(
            robot1, robot2, max_steps=1000
        )
        
        results['steps'].append(steps)
        results['exploration_ratios'].append(exploration_ratio)
        
        print(f"  Steps: {steps}")
        print(f"  Exploration: {exploration_ratio:.2%}")
    
    # 統計結果
    print("\n" + "=" * 60)
    print("測試結果統計")
    print("=" * 60)
    print(f"平均步數: {np.mean(results['steps']):.2f} ± {np.std(results['steps']):.2f}")
    print(f"平均探索率: {np.mean(results['exploration_ratios']):.2%}")
    print(f"最佳步數: {np.min(results['steps'])}")
    print(f"最差步數: {np.max(results['steps'])}")
    
    return results


def compare_with_dqn():
    """
    比較NeuralCoMapping和DQN的性能
    需要先運行你原本的DQN測試獲取baseline
    """
    print("\n" + "=" * 60)
    print("性能比較: NeuralCoMapping vs DQN")
    print("=" * 60)
    
    # 測試NeuralCoMapping
    print("\n1. 測試NeuralCoMapping...")
    ncm_results = test_neural_comapping(num_episodes=5)
    
    # 這裡你需要填入你的DQN baseline結果
    # 或者從你的測試腳本獲取
    dqn_avg_steps = None  # TODO: 填入DQN的平均步數
    
    if dqn_avg_steps is not None:
        ncm_avg_steps = np.mean(ncm_results['steps'])
        improvement = ((dqn_avg_steps - ncm_avg_steps) / dqn_avg_steps) * 100
        
        print("\n" + "=" * 60)
        print(f"DQN平均步數: {dqn_avg_steps:.2f}")
        print(f"NeuralCoMapping平均步數: {ncm_avg_steps:.2f}")
        print(f"改進: {improvement:+.2f}%")
        print("=" * 60)


if __name__ == "__main__":
    # 測試NeuralCoMapping
    results = test_neural_comapping(num_episodes=10)
    
    # 如果要比較性能
    # compare_with_dqn()