#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOC Distribution Analysis
Generates battery State of Charge distribution plots with safety boundaries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure fonts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

class SOCDistributionAnalysis:
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        
    def load_soc_data(self):
        """Load SOC trajectory data"""
        data_file = self.data_dir / "soc_trajectory.csv"
        if data_file.exists():
            data = pd.read_csv(data_file)
            print(f"Loaded SOC data: {len(data)} records")
            return data
        else:
            print(f"Error: SOC data file not found at {data_file}")
            return None
            
    def generate_distribution_plot(self, soc_data, language='en'):
        """Generate SOC distribution plot with safety boundaries"""
        plt.figure(figsize=(10, 6))
        
        if language == 'en':
            title = 'Battery SOC Distribution'
            xlabel = 'SOC (%)'
            ylabel = 'Frequency'
        else:
            title = '电池SOC分布'
            xlabel = 'SOC (%)'
            ylabel = '频次'
            
        # Plot distribution
        plt.hist(soc_data['soc'], bins=50, alpha=0.7, color='blue')
        
        # Add safety boundaries
        plt.axvline(x=20, color='red', linestyle='--', label='Safety Boundary')
        plt.axvline(x=90, color='red', linestyle='--')
        
        plt.title(title, fontsize=12)
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Save plot
        output_file = self.output_dir / f"soc_distribution_{language}.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved SOC distribution plot: {output_file}")
        
def main():
    data_dir = Path("data_processed")
    output_dir = Path("figures")
    
    analyzer = SOCDistributionAnalysis(data_dir, output_dir)
    soc_data = analyzer.load_soc_data()
    
    if soc_data is not None:
        analyzer.generate_distribution_plot(soc_data, 'en')
        analyzer.generate_distribution_plot(soc_data, 'zh')
    else:
        print("Error: Could not generate SOC distribution plots")

if __name__ == "__main__":
    main() 