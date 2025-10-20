#!/usr/bin/env python3
"""
图表生成工具
生成论文所需的图表和数据分析
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入模块
from generate_paper_figures import main as generate_paper_main
from generate_ieee_figures import main as generate_ieee_main
from real_vs_synthetic_comparison import RealSyntheticComparison
from code.plot_results import ResultsVisualizer, main as plot_results_main
from code.baseline_comparison import BaselineEvaluator, create_comparison_plots
from code.statistics import StatisticalAnalyzer

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ensure_directories():
    """创建输出目录"""
    directories = [
        "figures/paper/chinese",
        "figures/paper/english", 
        "figures/paper/ieee",
        "figures/experimental",
        "figures/validation",
        "figures/results"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"已创建目录: {dir_path}")


def generate_paper_figures():
    """生成论文图表（中英文版本）"""
    logger.info("正在生成论文图表...")
    
    try:
        # 调用原有的图表生成函数
        sys.argv = ['generate_paper_figures.py']  # 重置命令行参数
        generate_paper_main()
        logger.info("✓ 论文图表生成完成")
        return True
    except Exception as e:
        logger.error(f"✗ 论文图表生成失败: {e}")
        return False


def generate_ieee_figures():
    """生成IEEE格式图表"""
    logger.info("正在生成IEEE图表...")
    
    try:
        # 调用IEEE图表生成函数
        sys.argv = ['generate_ieee_figures.py']
        generate_ieee_main()
        logger.info("✓ IEEE图表生成完成")
        return True
    except Exception as e:
        logger.error(f"✗ IEEE图表生成失败: {e}")
        return False


def generate_comparison_figures():
    """生成数据对比图表"""
    logger.info("正在生成数据对比图表...")
    
    try:
        # 初始化对比分析器
        comparison = RealSyntheticComparison(output_dir='figures/paper/chinese')
        
        # 生成对比图表
        comparison.generate_real_vs_synthetic_comparison()
        logger.info("✓ 数据对比图表生成完成")
        return True
    except Exception as e:
        logger.error(f"✗ 数据对比图表生成失败: {e}")
        return False


def generate_results_figures(runs_dir="runs", summary_file="results/performance_summary.csv"):
    """生成结果分析图表"""
    logger.info("正在生成结果图表...")
    
    try:
        if not Path(runs_dir).exists():
            logger.warning(f"TensorBoard目录不存在: {runs_dir}，跳过训练曲线生成")
            return True
            
        # 初始化可视化器
        visualizer = ResultsVisualizer()
        
        # 加载TensorBoard数据
        tb_data = visualizer.load_tensorboard_data(runs_dir)
        
        if tb_data:
            # 生成训练曲线
            visualizer.create_training_curves(tb_data, "figures/results/training_curves.png")
            
        # 如果有性能摘要文件，生成性能对比图
        if Path(summary_file).exists():
            visualizer.create_performance_comparison(summary_file, "figures/results/performance_comparison.png")
            visualizer.create_box_plots(summary_file, "figures/results")
            
        # 生成时序分析图
        visualizer.create_time_series_analysis("figures/results")
        
        logger.info("✓ 结果图表生成完成")
        return True
    except Exception as e:
        logger.error(f"✗ 结果图表生成失败: {e}")
        return False


def generate_baseline_figures(load_data="data_processed/load_data.parquet",
                            price_data="data_processed/price_data.parquet", 
                            ev_data="data_processed/ev_demand.parquet",
                            soh_params="data_processed/soh_params.json"):
    """生成基线对比图表"""
    logger.info("正在生成基线对比图表...")
    
    try:
        # 检查数据文件是否存在
        required_files = [load_data, price_data, ev_data, soh_params]
        missing_files = [f for f in required_files if not Path(f).exists()]
        
        if missing_files:
            logger.warning(f"数据文件缺失，跳过基线对比: {missing_files}")
            return True
            
        # 初始化基线评估器
        evaluator = BaselineEvaluator(load_data, price_data, ev_data, soh_params)
        
        # 运行对比实验
        results = evaluator.compare_strategies(episodes=10)  # 快速测试用少量episode
        
        # 生成对比图表
        create_comparison_plots(results, "figures/results")
        
        logger.info("✓ 基线对比图表生成完成")
        return True
    except Exception as e:
        logger.error(f"✗ 基线对比图表生成失败: {e}")
        return False


def generate_statistical_figures(rl_dir="results/rl_results", 
                                baseline_dir="results/baseline_results"):
    """生成统计分析图表"""
    logger.info("正在生成统计分析图表...")
    
    try:
        # 检查结果目录是否存在
        if not Path(rl_dir).exists() or not Path(baseline_dir).exists():
            logger.warning("RL或基线结果目录不存在，跳过统计分析")
            return True
            
        # 初始化统计分析器
        analyzer = StatisticalAnalyzer()
        
        # 加载结果
        rl_results = analyzer.load_rl_results(rl_dir)
        baseline_results = analyzer.load_baseline_results(baseline_dir)
        
        # 计算统计量
        rl_stats = analyzer.calculate_rl_statistics(rl_results)
        
        # 执行显著性检验
        test_results = analyzer.perform_significance_tests(rl_stats, baseline_results)
        
        # 生成统计图表
        analyzer.create_statistical_plots(rl_stats, baseline_results, test_results, "figures/results")
        
        logger.info("✓ 统计分析图表生成完成")
        return True
    except Exception as e:
        logger.error(f"✗ 统计分析图表生成失败: {e}")
        return False


def create_figures_inventory():
    """创建图表清单文件"""
    logger.info("正在创建图表清单...")
    
    inventory_content = """# EVVES-DDDQN 图表清单

## 目录结构
```
figures/
├── paper/              # 论文图表
│   ├── chinese/        # 中文版本
│   ├── english/        # 英文版本  
│   └── ieee/           # IEEE格式
├── experimental/       # 实验图表
├── validation/         # 验证图表
└── results/           # 结果分析图表
```

## 图表说明

### 论文图表 (figures/paper/)
- **系统架构图**: fig1_system_architecture.pdf/png
- **负荷对比图**: fig2_load_profile_comparison.pdf/png  
- **训练曲线图**: fig3_training_curves.pdf/png
- **方法对比图**: fig4_method_comparison.pdf/png
- **鲁棒性分析**: fig5_robustness_analysis.pdf/png
- **V2G模式图**: fig6_v2g_pattern.pdf/png
- **复杂度分析**: fig7_complexity_analysis.pdf/png
- **敏感性分析**: fig8_sensitivity_analysis.pdf/png

### IEEE图表 (figures/paper/ieee/)
- **训练曲线**: fig1_training_curves.pdf/png
- **基线对比**: fig2_baseline_comparison.pdf/png
- **系统概览**: fig3_system_overview.pdf/png

### 结果图表 (figures/results/)
- **训练曲线**: training_curves.png
- **性能对比**: performance_comparison.png
- **成本箱线图**: cost_boxplot.png
- **SOH箱线图**: soh_boxplot.png
- **时序分析**: soh_time_series.png
- **基线对比**: baseline_comparison.png
- **统计分析**: statistical_analysis.png

### 验证图表 (figures/validation/)
- **PDF格式**: pdf/
- **PNG格式**: png/

## 生成方式
所有图表均由 `scripts/generate_all_figures.py` 统一生成。

### 使用方法
```bash
# 生成所有图表
python scripts/generate_all_figures.py --all

# 仅生成论文图表  
python scripts/generate_all_figures.py --paper

# 仅生成结果图表
python scripts/generate_all_figures.py --results
```
"""
    
    with open("FIGURES_INVENTORY.md", "w", encoding="utf-8") as f:
        f.write(inventory_content)
    
    logger.info("✓ 图表清单已创建: FIGURES_INVENTORY.md")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成EVVES-DDDQN项目的所有图表")
    
    parser.add_argument("--all", action="store_true", help="生成所有图表")
    parser.add_argument("--paper", action="store_true", help="仅生成论文图表")
    parser.add_argument("--ieee", action="store_true", help="仅生成IEEE图表")
    parser.add_argument("--results", action="store_true", help="仅生成结果图表")
    parser.add_argument("--baseline", action="store_true", help="仅生成基线对比图表")
    parser.add_argument("--statistics", action="store_true", help="仅生成统计分析图表")
    parser.add_argument("--comparison", action="store_true", help="仅生成数据对比图表")
    
    # 数据路径参数
    parser.add_argument("--runs_dir", default="runs", help="TensorBoard运行目录")
    parser.add_argument("--summary_file", default="results/performance_summary.csv", help="性能摘要文件")
    parser.add_argument("--load_data", default="data_processed/load_data.parquet", help="负荷数据文件")
    parser.add_argument("--price_data", default="data_processed/price_data.parquet", help="价格数据文件")
    parser.add_argument("--ev_data", default="data_processed/ev_demand.parquet", help="EV需求数据文件")
    parser.add_argument("--soh_params", default="data_processed/soh_params.json", help="SOH参数文件")
    parser.add_argument("--rl_dir", default="results/rl_results", help="RL结果目录")
    parser.add_argument("--baseline_dir", default="results/baseline_results", help="基线结果目录")
    
    args = parser.parse_args()
    
    # 如果没有指定任何选项，默认生成所有图表
    if not any([args.all, args.paper, args.ieee, args.results, args.baseline, 
                args.statistics, args.comparison]):
        args.all = True
    
    # 确保目录存在
    ensure_directories()
    
    success_count = 0
    total_count = 0
    
    # 根据参数选择生成哪些图表
    if args.all or args.paper:
        total_count += 1
        if generate_paper_figures():
            success_count += 1
            
    if args.all or args.ieee:
        total_count += 1
        if generate_ieee_figures():
            success_count += 1
            
    if args.all or args.comparison:
        total_count += 1
        if generate_comparison_figures():
            success_count += 1
            
    if args.all or args.results:
        total_count += 1
        if generate_results_figures(args.runs_dir, args.summary_file):
            success_count += 1
            
    if args.all or args.baseline:
        total_count += 1
        if generate_baseline_figures(args.load_data, args.price_data, args.ev_data, args.soh_params):
            success_count += 1
            
    if args.all or args.statistics:
        total_count += 1
        if generate_statistical_figures(args.rl_dir, args.baseline_dir):
            success_count += 1
    
    # 创建图表清单
    create_figures_inventory()
    
    # 输出总结
    logger.info(f"\n图表生成完成: {success_count}/{total_count} 成功")
    
    if success_count == total_count:
        logger.info("所有图表生成成功")
        logger.info("图表保存位置: figures/ 目录")
        logger.info("详细清单: FIGURES_INVENTORY.md")
        return True
    else:
        logger.warning(f"部分图表生成失败 ({total_count - success_count} 个)")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 