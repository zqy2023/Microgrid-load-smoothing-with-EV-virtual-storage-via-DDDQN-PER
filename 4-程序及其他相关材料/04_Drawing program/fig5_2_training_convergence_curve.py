#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图5.2：DDDQN-PER训练收敛曲线生成脚本（双Y轴，横坐标对齐）
基于强化学习的电动汽车虚拟储能负荷平滑算法研究

- DDDQN-PER和Baseline-DQN奖励曲线分别画在双Y轴上，横坐标长度一致
- 收敛点标注在DDDQN-PER曲线上
- 输出中英文PNG图，风格与图5.1一致
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 设置中文字体
font_path = "C:/Windows/Fonts/simsun.ttc"
if os.path.exists(font_path):
    zh_font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False
else:
    zh_font = None

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# 读取数据
DDDQN_EPISODE_PATH = '../dddqn_per_july3/seed_0/episode_data.csv'
BASELINE_EPISODE_PATH = '../baseline_dqn_july9/seed_0/episode_data.csv'

dddqn_df = pd.read_csv(DDDQN_EPISODE_PATH)
dddqn_rewards = dddqn_df['reward'].values

baseline_df = pd.read_csv(BASELINE_EPISODE_PATH)
baseline_rewards = baseline_df['reward'].values

# 横坐标对齐
max_len = max(len(dddqn_rewards), len(baseline_rewards))
dddqn_x = np.arange(max_len)
baseline_x = np.arange(max_len)
dddqn_y = np.full(max_len, np.nan)
baseline_y = np.full(max_len, np.nan)
dddqn_y[:len(dddqn_rewards)] = dddqn_rewards
baseline_y[:len(baseline_rewards)] = baseline_rewards

def detect_converge_point(data, window=20, threshold_ratio=0.01):
    ma = np.convolve(data, np.ones(window)/window, mode='valid')
    threshold = threshold_ratio * np.max(np.abs(ma))
    for i in range(window, len(ma)):
        if np.max(np.abs(ma[i-window:i] - ma[i])) < threshold:
            return i + window//2
    return np.argmax(ma) + window//2

window = 20
converge_idx = detect_converge_point(dddqn_rewards, window=window)

# ----------- 中文版 -----------
def plot_chinese():
    fig, ax1 = plt.subplots(figsize=(16, 8))
    color1 = '#d62728'
    color2 = '#1f77b4'
    l1 = ax1.plot(dddqn_x, dddqn_y, color=color1, linewidth=2, label='DDDQN-PER奖励')
    ax1.set_xlabel('训练回合 (Episode)', fontsize=24, fontproperties=zh_font)
    ax1.set_ylabel('DDDQN-PER奖励', fontsize=24, color=color1, fontproperties=zh_font)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=22)
    ax1.tick_params(axis='x', labelsize=22)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    # 收敛点
    ax1.axvline(x=converge_idx, color='blue', linestyle='--', linewidth=2, label='收敛点')
    y_converge = dddqn_rewards[converge_idx] if converge_idx < len(dddqn_rewards) else np.nan
    ax1.annotate('收敛点', xy=(converge_idx, y_converge), xytext=(converge_idx+10, y_converge+0.05*abs(y_converge)),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5), fontsize=20, color='blue', fontweight='bold', fontproperties=zh_font)
    # Baseline-DQN右Y轴
    ax2 = ax1.twinx()
    l2 = ax2.plot(baseline_x, baseline_y, color=color2, linewidth=2, label='Baseline-DQN奖励')
    ax2.set_ylabel('Baseline-DQN奖励', fontsize=24, color=color2, fontproperties=zh_font)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=22)
    # 合并图例
    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=22, frameon=True, fancybox=True, shadow=True, framealpha=0.9, prop=zh_font)
    fig.tight_layout()
    plt.savefig('图5.2_DDDQN-PER训练收敛曲线_中文.png', dpi=300, bbox_inches='tight')
    plt.savefig('图5.2_DDDQN-PER训练收敛曲线_中文.pdf', bbox_inches='tight')
    plt.show()

# ----------- 英文版 -----------
def plot_english():
    fig, ax1 = plt.subplots(figsize=(16, 8))
    color1 = '#d62728'
    color2 = '#1f77b4'
    l1 = ax1.plot(dddqn_x, dddqn_y, color=color1, linewidth=2, label='DDDQN-PER Reward')
    ax1.set_xlabel('Episode', fontsize=24, fontfamily='Times New Roman')
    ax1.set_ylabel('DDDQN-PER Reward', fontsize=24, color=color1, fontfamily='Times New Roman')
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=22)
    ax1.tick_params(axis='x', labelsize=22)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    # 收敛点
    ax1.axvline(x=converge_idx, color='blue', linestyle='--', linewidth=2, label='Convergence Point')
    y_converge = dddqn_rewards[converge_idx] if converge_idx < len(dddqn_rewards) else np.nan
    ax1.annotate('Convergence', xy=(converge_idx, y_converge), xytext=(converge_idx+10, y_converge+0.05*abs(y_converge)),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5), fontsize=20, color='blue', fontweight='bold')
    # Baseline-DQN右Y轴
    ax2 = ax1.twinx()
    l2 = ax2.plot(baseline_x, baseline_y, color=color2, linewidth=2, label='Baseline-DQN Reward')
    ax2.set_ylabel('Baseline-DQN Reward', fontsize=24, color=color2, fontfamily='Times New Roman')
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=22)
    # 合并图例
    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=20, frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    fig.tight_layout()
    plt.savefig('Figure_5_2_DDDQN-PER_Training_Convergence_Curve_English.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure_5_2_DDDQN-PER_Training_Convergence_Curve_English.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("生成图5.2：DDDQN-PER训练收敛曲线...")
    print("生成中文版...")
    plot_chinese()
    print("生成英文版...")
    plot_english()
    print("图5.2生成完成！") 