#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traditional Strategy Analysis
Analyzes SOC trajectories of traditional control strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure fonts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

class TraditionalStrategyAnalysis:
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        
    def load_strategy_data(self):
        """Load traditional strategy data"""
        results_file = self.data_dir / "traditional_strategies_results.csv"
        summary_file = self.data_dir / "strategies_summary.csv"
        raw_data_file = self.data_dir / "all_strategies_raw_data.json"
        
        data = {}
        if all(f.exists() for f in [results_file, summary_file, raw_data_file]):
            data['results'] = pd.read_csv(results_file)
            data['summary'] = pd.read_csv(summary_file)
            with open(raw_data_file, 'r') as f:
                data['raw_data'] = json.load(f)
            print("Loaded all strategy data files")
            return data
        else:
            print("Error: Some strategy data files are missing")
            return None
            
    def analyze_soc_trajectories(self, data):
        """Analyze SOC trajectories of different strategies"""
        strategies = data['results']['strategy'].unique()
        analysis = {}
        
        for strategy in strategies:
            strategy_data = data['results'][data['results']['strategy'] == strategy]
            analysis[strategy] = {
                'mean_soc': strategy_data['soc'].mean(),
                'std_soc': strategy_data['soc'].std(),
                'min_soc': strategy_data['soc'].min(),
                'max_soc': strategy_data['soc'].max(),
                'violation_rate': len(strategy_data[
                    (strategy_data['soc'] < 20) | (strategy_data['soc'] > 90)
                ]) / len(strategy_data)
            }
            
        return analysis
        
    def generate_comparison_plot(self, data, analysis, language='en'):
        """Generate strategy comparison plot"""
        plt.figure(figsize=(12, 8))
        
        if language == 'en':
            title = 'Traditional Strategy SOC Analysis'
            xlabel = 'Strategy'
            ylabel = 'SOC Statistics (%)'
            legend_labels = ['Mean SOC', 'SOC Range']
        else:
            title = '传统策略SOC分析'
            xlabel = '策略'
            ylabel = 'SOC统计 (%)'
            legend_labels = ['平均SOC', 'SOC范围']
            
        strategies = list(analysis.keys())
        mean_soc = [analysis[s]['mean_soc'] for s in strategies]
        min_soc = [analysis[s]['min_soc'] for s in strategies]
        max_soc = [analysis[s]['max_soc'] for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        plt.bar(x, mean_soc, width, label=legend_labels[0])
        plt.vlines(x, min_soc, max_soc, color='r', linestyle='-', label=legend_labels[1])
        
        plt.axhline(y=20, color='r', linestyle='--')
        plt.axhline(y=90, color='r', linestyle='--')
        
        plt.title(title, fontsize=12)
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        plt.xticks(x, strategies, rotation=45)
        plt.legend()
        plt.grid(True)
        
        # Save plot
        output_file = self.output_dir / f"traditional_strategy_analysis_{language}.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved strategy analysis plot: {output_file}")
        
    def save_analysis_results(self, analysis):
        """Save analysis results"""
        output_file = self.output_dir / "traditional_strategy_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Saved analysis results: {output_file}")
        
def main():
    data_dir = Path("data_processed")
    output_dir = Path("figures")
    
    analyzer = TraditionalStrategyAnalysis(data_dir, output_dir)
    data = analyzer.load_strategy_data()
    
    if data is not None:
        analysis = analyzer.analyze_soc_trajectories(data)
        analyzer.generate_comparison_plot(data, analysis, 'en')
        analyzer.generate_comparison_plot(data, analysis, 'zh')
        analyzer.save_analysis_results(analysis)
    else:
        print("Error: Could not perform traditional strategy analysis")

if __name__ == "__main__":
    main() 