#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图5.2：DDDQN-PER vs Baseline DQN训练收敛曲线
使用完整训练数据生成收敛曲线对比
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from pathlib import Path

class DDDQNConvergencePlotter:
    def __init__(self):
        self.figure_dir = Path('comprehensive_results/figure')
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        
        self.dddqn_data_path = 'results/train_seed_0_20250703_054947/episode_data.csv'
        self.baseline_data_path = 'results/train_seed_0_20250709_133846/episode_data.csv'
        
    def load_training_data(self):
        """加载DDDQN-PER和Baseline DQN训练数据"""
        print(f"加载DDDQN-PER训练数据: {self.dddqn_data_path}")
        print(f"加载Baseline DQN训练数据: {self.baseline_data_path}")
        
        dddqn_df = pd.read_csv(self.dddqn_data_path)
        baseline_df = pd.read_csv(self.baseline_data_path)
        
        print(f"DDDQN-PER训练数据形状: {dddqn_df.shape}")
        print(f"Baseline DQN训练数据形状: {baseline_df.shape}")
        print(f"DDDQN-PER Episode范围: {dddqn_df['episode'].min()} - {dddqn_df['episode'].max()}")
        print(f"Baseline DQN Episode范围: {baseline_df['episode'].min()} - {baseline_df['episode'].max()}")
        print(f"DDDQN-PER奖励范围: {dddqn_df['reward'].min():.2f} - {dddqn_df['reward'].max():.2f}")
        print(f"Baseline DQN奖励范围: {baseline_df['reward'].min():.2f} - {baseline_df['reward'].max():.2f}")
        
        min_episodes = min(len(dddqn_df), len(baseline_df))
        print(f"\n使用前{min_episodes}个episode进行对比")
        
        return dddqn_df.head(min_episodes), baseline_df.head(min_episodes)
    
    def find_convergence_point(self, rewards, episodes, threshold_ratio=0.1):
        """找到收敛点：基于论文4.5节的收敛性判定标准"""
        # 论文中的收敛性判定标准：平均奖励波动率控制
        window_size = 500  # 论文中设定的Δt = 500步
        volatility_threshold = 0.005  # 论文中设定的εr = 0.005
        
        if len(rewards) > window_size:
            # 计算移动平均奖励
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            moving_avg_episodes = episodes[window_size-1:]
            
            # 计算奖励波动率：|Ri - Ri-1|的平均值
            reward_volatility = []
            for i in range(1, len(moving_avg)):
                volatility = abs(moving_avg[i] - moving_avg[i-1])
                reward_volatility.append(volatility)
            
            # 找到第一个波动率低于阈值的点
            convergence_idx = np.where(np.array(reward_volatility) < volatility_threshold)[0]
            if len(convergence_idx) > 0:
                # 返回对应的episode和奖励值
                conv_episode = moving_avg_episodes[convergence_idx[0] + 1]  # +1因为volatility少一个点
                conv_reward = moving_avg[convergence_idx[0] + 1]
                return conv_episode, conv_reward
        
        # 如果没找到，使用备选方案：奖励开始稳定改善的点
        window_size = 50
        if len(rewards) > window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            initial_avg = np.mean(rewards[:100])
            improvement_threshold = initial_avg + abs(initial_avg) * threshold_ratio
            convergence_idx = np.where(moving_avg > improvement_threshold)[0]
            if len(convergence_idx) > 0:
                return episodes[window_size-1 + convergence_idx[0]], moving_avg[convergence_idx[0]]
        
        # 如果都没找到，返回None
        return None, None
    
    def plot_convergence_curve(self, dddqn_df, baseline_df, language='english'):
        """绘制DDDQN-PER vs Baseline DQN训练收敛曲线"""
        # 根据语言设置字体
        if language == 'chinese':
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        else:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax2 = ax1.twinx()  # 创建双Y轴
        
        # DDDQN-PER数据
        dddqn_episodes = dddqn_df['episode'].values
        dddqn_rewards = dddqn_df['reward'].values
        
        # Baseline DQN数据
        baseline_episodes = baseline_df['episode'].values
        baseline_rewards = baseline_df['reward'].values
        
        # 根据语言设置标签
        if language == 'chinese':
            dddqn_label = 'DDDQN-PER训练奖励'
            baseline_label = 'Baseline DQN训练奖励'
            xlabel = '训练Episode'
            dddqn_ylabel = 'DDDQN-PER Episode奖励'
            baseline_ylabel = 'Baseline DQN Episode奖励'
            data_source = '数据来源: 完整DDDQN-PER和Baseline DQN训练结果'
        else:
            dddqn_label = 'DDDQN-PER Training Reward'
            baseline_label = 'Baseline DQN Training Reward'
            xlabel = 'Training Episode'
            dddqn_ylabel = 'DDDQN-PER Episode Reward'
            baseline_ylabel = 'Baseline DQN Episode Reward'
            data_source = 'Data Source: Complete DDDQN-PER and Baseline DQN Training Results'
        
        # 绘制DDDQN-PER曲线（左Y轴）
        line1 = ax1.plot(dddqn_episodes, dddqn_rewards, 'b-', linewidth=1.5, alpha=0.7, label=dddqn_label)
        
        # 绘制Baseline DQN曲线（右Y轴）
        line2 = ax2.plot(baseline_episodes, baseline_rewards, 'r-', linewidth=1.5, alpha=0.7, label=baseline_label)
        
        # 计算并绘制滑动平均
        window_size = 50
        
        # DDDQN-PER滑动平均
        if len(dddqn_rewards) > window_size:
            dddqn_moving_avg = np.convolve(dddqn_rewards, np.ones(window_size)/window_size, mode='valid')
            dddqn_moving_avg_episodes = dddqn_episodes[window_size-1:]
            ax1.plot(dddqn_moving_avg_episodes, dddqn_moving_avg, 'b-', linewidth=2.5, alpha=0.8)
        
        # Baseline DQN滑动平均
        if len(baseline_rewards) > window_size:
            baseline_moving_avg = np.convolve(baseline_rewards, np.ones(window_size)/window_size, mode='valid')
            baseline_moving_avg_episodes = baseline_episodes[window_size-1:]
            ax2.plot(baseline_moving_avg_episodes, baseline_moving_avg, 'r-', linewidth=2.5, alpha=0.8)
        
        # 找到并标注收敛点
        dddqn_conv_episode, dddqn_conv_reward = self.find_convergence_point(dddqn_rewards, dddqn_episodes)
        baseline_conv_episode, baseline_conv_reward = self.find_convergence_point(baseline_rewards, baseline_episodes)
        
        if dddqn_conv_episode is not None:
            # 标注DDDQN-PER收敛点
            ax1.axvline(x=dddqn_conv_episode, color='blue', linestyle='--', linewidth=2, alpha=0.8)
            # 在移动平均线上标注收敛点
            ax1.scatter(dddqn_conv_episode, dddqn_conv_reward, color='blue', s=100, zorder=5)
            
            # 添加收敛点文本标注
            if language == 'chinese':
                conv_text = f'DDDQN-PER收敛点\nEpisode {dddqn_conv_episode}'
            else:
                conv_text = f'DDDQN-PER Convergence\nEpisode {dddqn_conv_episode}'
            
            ax1.annotate(conv_text, 
                        xy=(dddqn_conv_episode, dddqn_conv_reward),
                        xytext=(dddqn_conv_episode - 300, dddqn_conv_reward + 65),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2),
                        fontsize=10, color='black', weight='bold')
        
        if baseline_conv_episode is not None:
            # 标注Baseline DQN收敛点
            ax2.axvline(x=baseline_conv_episode, color='red', linestyle='--', linewidth=2, alpha=0.8)
            # 在移动平均线上标注收敛点
            ax2.scatter(baseline_conv_episode, baseline_conv_reward, color='red', s=100, zorder=5)
            
            # 添加收敛点文本标注
            if language == 'chinese':
                conv_text = f'Baseline DQN收敛点\nEpisode {baseline_conv_episode}'
            else:
                conv_text = f'Baseline DQN Convergence\nEpisode {baseline_conv_episode}'
            
            ax2.annotate(conv_text, 
                        xy=(baseline_conv_episode, baseline_conv_reward),
                        xytext=(baseline_conv_episode + 300, baseline_conv_reward + 40),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2),
                        fontsize=10, color='black', weight='bold')
        
        # 设置图表属性
        ax1.set_xlabel(xlabel, fontsize=16, fontweight='bold')
        ax1.set_ylabel(dddqn_ylabel, fontsize=16, fontweight='bold', color='blue')
        ax2.set_ylabel(baseline_ylabel, fontsize=16, fontweight='bold', color='red')
        
        # 设置Y轴颜色
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # 设置网格
        ax1.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, fontsize=12, loc='upper right')
        
        # 设置坐标轴范围
        max_episodes = max(dddqn_episodes[-1], baseline_episodes[-1])
        ax1.set_xlim(0, max_episodes)
        
        # 添加数据来源标注
        ax1.text(0.02, 0.02, data_source, 
                transform=ax1.transAxes, fontsize=10, style='italic',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def generate_figures(self):
        """生成中英文版本的图5.2"""
        # 加载数据
        dddqn_df, baseline_df = self.load_training_data()
        
        # 生成中文版本
        fig = self.plot_convergence_curve(dddqn_df, baseline_df, language='chinese')
        fig.savefig(self.figure_dir / '图5.2_DDDQN-PER训练收敛曲线_中文.png', 
                   dpi=300, bbox_inches='tight')
        fig.savefig(self.figure_dir / '图5.2_DDDQN-PER训练收敛曲线_中文.pdf', 
                   bbox_inches='tight')
        plt.close(fig)
        
        # 生成英文版本
        fig = self.plot_convergence_curve(dddqn_df, baseline_df, language='english')
        fig.savefig(self.figure_dir / 'Figure_5_2_DDDQN-PER_Training_Convergence_Curve_English.png', 
                   dpi=300, bbox_inches='tight')
        fig.savefig(self.figure_dir / 'Figure_5_2_DDDQN-PER_Training_Convergence_Curve_English.pdf', 
                   bbox_inches='tight')
        plt.close(fig)
        
        print("图5.2 DDDQN-PER vs Baseline DQN训练收敛曲线对比生成完成！")
        print(f"文件保存位置: {self.figure_dir}")

def main():
    plotter = DDDQNConvergencePlotter()
    plotter.generate_figures()

if __name__ == "__main__":
    main() 