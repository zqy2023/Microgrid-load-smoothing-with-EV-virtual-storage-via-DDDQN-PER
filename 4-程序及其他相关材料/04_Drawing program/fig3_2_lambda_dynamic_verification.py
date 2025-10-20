#!/usr/bin/env python3
"""
λ(t)动态验证分析脚本
分析价格归一化因子λ(t)的动态特性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import seaborn as sns

# 配置绘图参数
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

class LambdaDynamicAnalyzer:
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置时间窗口参数
        self.window_size = 672  # 7天 * 96(15分钟间隔)
        
    def load_price_data(self):
        """加载电价数据"""
        price_file = self.data_dir / "price_15min_fullyear.parquet"
        if price_file.exists():
            data = pd.read_parquet(price_file)
            print(f"已加载电价数据: {len(data)}条记录")
            return data
        else:
            print(f"错误: 未找到电价数据文件 {price_file}")
            return None
            
    def calculate_lambda(self, price_data):
        """计算λ(t)时间序列"""
        # 计算滚动最大价格
        rolling_max = price_data['price'].rolling(window=self.window_size, min_periods=1).max()
        
        # 计算λ(t)
        lambda_t = price_data['price'] / rolling_max
        lambda_t = lambda_t.clip(0, 1)  # 限制在[0,1]范围内
        
        return lambda_t
    
    def analyze_lambda_dynamics(self, lambda_series):
        """分析λ(t)的动态特性"""
        stats = {
            'mean': lambda_series.mean(),
            'std': lambda_series.std(),
            'min': lambda_series.min(),
            'max': lambda_series.max(),
            'median': lambda_series.median(),
            'q25': lambda_series.quantile(0.25),
            'q75': lambda_series.quantile(0.75)
        }
        
        # 计算自相关性
        acf = pd.Series(lambda_series).autocorr(lag=96)  # 24小时lag
        stats['daily_autocorr'] = acf
        
        return stats
        
    def plot_lambda_verification(self, price_data, lambda_series, language='zh'):
        """绘制λ(t)验证图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 绘制一周数据示例
        week_data = price_data.iloc[:672]  # 一周数据
        week_lambda = lambda_series.iloc[:672]
        
        # 上图：原始电价和滚动最大值
        ax1.plot(week_data.index, week_data['price'], label='实时电价' if language=='zh' else 'Real-time Price')
        ax1.plot(week_data.index, week_data['price'].rolling(window=self.window_size, min_periods=1).max(),
                 label='滚动最大值' if language=='zh' else 'Rolling Maximum')
        
        if language == 'zh':
            ax1.set_title('电价动态特征')
            ax1.set_xlabel('时间')
            ax1.set_ylabel('电价 (元/kWh)')
        else:
            ax1.set_title('Price Dynamics')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Price (CNY/kWh)')
        ax1.legend()
        ax1.grid(True)
        
        # 下图：λ(t)时间序列
        ax2.plot(week_lambda.index, week_lambda, color='red')
        if language == 'zh':
            ax2.set_title('λ(t)动态变化')
            ax2.set_xlabel('时间')
            ax2.set_ylabel('λ(t)值')
        else:
            ax2.set_title('λ(t) Dynamics')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('λ(t) Value')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        output_file = self.output_dir / f"lambda_dynamics_{language}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已生成λ(t)动态验证图: {output_file}")
        
    def plot_lambda_distribution(self, lambda_series, language='zh'):
        """绘制λ(t)分布图"""
        plt.figure(figsize=(10, 6))
        
        # 绘制分布直方图
        sns.histplot(lambda_series, bins=50, kde=True)
        
        if language == 'zh':
            plt.title('λ(t)值分布')
            plt.xlabel('λ(t)值')
            plt.ylabel('频次')
        else:
            plt.title('λ(t) Value Distribution')
            plt.xlabel('λ(t) Value')
            plt.ylabel('Frequency')
            
        plt.grid(True)
        
        # 保存图片
        output_file = self.output_dir / f"lambda_distribution_{language}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已生成λ(t)分布图: {output_file}")

def main():
    """主函数"""
    data_dir = Path("data_processed")
    output_dir = Path("figures")
    
    analyzer = LambdaDynamicAnalyzer(data_dir, output_dir)
        
    # 加载数据
    price_data = analyzer.load_price_data()
    if price_data is None:
        return
        
    # 计算λ(t)
    lambda_series = analyzer.calculate_lambda(price_data)
        
    # 分析动态特性
    stats = analyzer.analyze_lambda_dynamics(lambda_series)
        
    # 生成验证图
    analyzer.plot_lambda_verification(price_data, lambda_series, 'zh')
    analyzer.plot_lambda_verification(price_data, lambda_series, 'en')
    
    # 生成分布图
    analyzer.plot_lambda_distribution(lambda_series, 'zh')
    analyzer.plot_lambda_distribution(lambda_series, 'en')
    
    # 保存统计结果
    with open(output_dir / "lambda_statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("λ(t)动态验证分析完成")

if __name__ == "__main__":
    main() 
 
 
 
 
 
 
 
 