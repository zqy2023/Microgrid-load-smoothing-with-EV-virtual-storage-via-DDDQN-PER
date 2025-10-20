#!/usr/bin/env python3
"""
消融实验分析
评估DDDQN-PER算法各组件的贡献
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import seaborn as sns

# 绘图配置
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

class AblationAnalyzer:
    def __init__(self, results_dir, output_dir):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 算法变体
        self.variants = {
            'DDDQN-PER': '完整模型',
            'DQN': '基础DQN',
            'DDQN': '双DQN',
            'DDDQN': '双延迟DQN',
            'DDDQN-Basic': '无优先经验回放'
        }
        
    def load_results(self):
        """加载实验结果"""
        results = {}
        for variant in self.variants:
            result_file = self.results_dir / f"{variant.lower()}_results.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    results[variant] = json.load(f)
            else:
                print(f"警告: 未找到{variant}的结果文件")
        return results
        
    def analyze_performance(self, results):
        """分析各变体的性能指标"""
        metrics = {
            'reward': '累积奖励',
            'cost': '总成本',
            'soh': '电池健康度',
            'convergence': '收敛速度'
        }
        
        performance = {}
        for variant, data in results.items():
            performance[variant] = {
                'reward_mean': np.mean(data['rewards']),
                'reward_std': np.std(data['rewards']),
                'cost_mean': np.mean(data['costs']),
                'cost_std': np.std(data['costs']),
                'soh_mean': np.mean(data['soh_values']),
                'soh_std': np.std(data['soh_values']),
                'convergence': data['convergence_step']
            }
        return performance
        
    def plot_ablation_results(self, performance, language='zh'):
        """绘制消融实验结果对比图"""
        metrics = ['reward', 'cost', 'soh', 'convergence']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx//2, idx%2]
            means = [performance[v][f'{metric}_mean'] for v in self.variants]
            stds = [performance[v][f'{metric}_std'] for v in self.variants]
            
            # 绘制条形图
            ax.bar(range(len(self.variants)), means, yerr=stds, capsize=5)
            ax.set_xticks(range(len(self.variants)))
            ax.set_xticklabels(self.variants, rotation=45)
            
            # 设置标题和标签
            if language == 'zh':
                metric_names = {
                    'reward': '累积奖励',
                    'cost': '总成本(元)',
                    'soh': '电池健康度(%)',
                    'convergence': '收敛步数'
                }
                ax.set_title(metric_names[metric])
            else:
                metric_names = {
                    'reward': 'Cumulative Reward',
                    'cost': 'Total Cost (CNY)',
                    'soh': 'Battery Health (%)',
                    'convergence': 'Convergence Steps'
                }
                ax.set_title(metric_names[metric])
            
            ax.grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        output_file = self.output_dir / f"ablation_analysis_{language}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已生成消融实验分析图: {output_file}")
        
    def generate_summary_table(self, performance):
        """生成消融实验结果汇总表"""
        summary = pd.DataFrame(performance).T
        summary.index.name = 'Variant'
        
        # 保存结果
        output_file = self.output_dir / "ablation_summary.csv"
        summary.to_csv(output_file)
        print(f"已生成结果汇总表: {output_file}")
        
        return summary

def main():
    """主函数"""
    results_dir = Path("results/ablation")
    output_dir = Path("figures/ablation")
    
    analyzer = AblationAnalyzer(results_dir, output_dir)
    
    # 加载并分析结果
    results = analyzer.load_results()
    performance = analyzer.analyze_performance(results)
    
    # 生成图表和汇总
    analyzer.plot_ablation_results(performance, 'zh')
    analyzer.plot_ablation_results(performance, 'en')
    summary = analyzer.generate_summary_table(performance)
    
    print("消融实验分析完成")

if __name__ == "__main__":
    main() 
 
 
 
 
 
 
 
 