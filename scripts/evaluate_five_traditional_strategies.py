#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5种传统策略评估
使用与DDDQN、Baseline DQN、Rule-Based相同的环境和数据进行评估
"""

import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ToUAgent:
    """分时电价策略 - 基于时间段的充电决策"""
    def __init__(self):
        # 澳洲典型分时电价时段 (小时)
        self.peak_hours = [17, 18, 19, 20]  # 高峰期
        self.off_peak_hours = [0, 1, 2, 3, 4, 5, 22, 23]  # 低谷期
        
    def get_action(self, obs):
        # 从时间进度计算当前小时
        time_progress = obs[4]  # 时间进度 [0,1]
        hour = int((time_progress * 8760) % 24)  # 转换为24小时制
        soc = obs[3]
        
        if soc < 0.2:
            return 1  # 安全充电
        elif soc > 0.9:
            return 0  # 停止充电
        elif hour in self.off_peak_hours:
            return 1  # 低谷期充电
        elif hour in self.peak_hours:
            return 0  # 高峰期停止
        else:
            return 1  # 平时期充电

class SafeBotAgent:
    """安全机器人策略 - 保守充电策略"""
    def __init__(self, safe_soc_min=0.3, safe_soc_max=0.8):
        self.safe_soc_min = safe_soc_min
        self.safe_soc_max = safe_soc_max
        
    def get_action(self, obs):
        soc = obs[3]
        price = obs[0]
        
        # 保守策略：维持安全SOC范围
        if soc < self.safe_soc_min:
            return 1  # 低于安全下限，必须充电
        elif soc > self.safe_soc_max:
            return 0  # 高于安全上限，停止充电
        elif price < 25.0:  # 低价时充电
            return 1
        else:
            return 0

class MPCAgent:
    """模型预测控制策略 - 基于价格预测的优化"""
    def __init__(self, prediction_horizon=24):
        self.prediction_horizon = prediction_horizon
        self.price_history = []
        
    def get_action(self, obs):
        price = obs[0]
        soc = obs[3]
        
        # 维护价格历史
        self.price_history.append(price)
        if len(self.price_history) > 96:  # 保持4天历史
            self.price_history.pop(0)
            
        if soc < 0.2:
            return 1  # 安全充电
        elif soc > 0.9:
            return 0  # 停止充电
        elif len(self.price_history) < 24:
            return 1 if price < 30 else 0  # 初期简单策略
        else:
            # 预测策略：如果当前价格低于历史平均值，充电
            recent_avg = np.mean(self.price_history[-24:])
            return 1 if price < recent_avg * 0.9 else 0

class GreedyPriceAgent:
    """贪婪价格策略 - 纯粹基于价格的贪婪决策"""
    def __init__(self, price_threshold=20.0):
        self.price_threshold = price_threshold
        
    def get_action(self, obs):
        price = obs[0]
        soc = obs[3]
        
        if soc < 0.15:
            return 1  # 紧急充电
        elif soc > 0.95:
            return 0  # 满电停止
        else:
            # 贪婪策略：只在极低价格时充电
            return 1 if price < self.price_threshold else 0

class RandomAgent:
    """随机策略 - 作为基准对照"""
    def __init__(self, charge_probability=0.3):
        self.charge_probability = charge_probability
        
    def get_action(self, obs):
        soc = obs[3]
        
        if soc < 0.1:
            return 1  # 防止过放
        elif soc > 0.95:
            return 0  # 防止过充
        else:
            # 随机决策
            return 1 if np.random.random() < self.charge_probability else 0

class EVVESEnv:
    """标准EVVES环境 - 与RL训练完全一致"""
    def __init__(self, load_data, price_data, ev_data, soh_params):
        # 处理数据格式 - 确保使用真实数据
        self.load_data = self._extract_values(load_data, "负荷")
        self.price_data = self._extract_values(price_data, "价格") 
        self.ev_data = self._extract_values(ev_data, "EV需求")
        self.soh_params = soh_params
        
        logger.info(f"环境初始化: {len(self.price_data)}个时间点 (100%真实澳洲数据)")
        
        # 环境参数 - 与RL训练一致
        self.battery_capacity = 100.0
        self.max_charge_rate = 22.0
        self.dt = 0.25
        self.efficiency = 0.9
        self.reset()
        
    def _extract_values(self, data, data_type):
        """提取真实数值数据"""
        if isinstance(data, pd.DataFrame):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                values = data[numeric_cols[0]].values.astype(float)
                logger.info(f"{data_type}数据: 列'{numeric_cols[0]}', 长度{len(values)}")
                return values
        elif isinstance(data, pd.Series):
            return data.values.astype(float)
        return np.array(data, dtype=float).flatten()
        
    def reset(self):
        self.timestep = 0
        self.soc = 0.5
        self.soh = 1.0
        self.total_cost = 0.0
        self.total_energy = 0.0
        return self._get_obs()
    
    def _get_obs(self):
        if self.timestep >= len(self.price_data):
            return np.zeros(6)
        return np.array([
            self.price_data[self.timestep],           # 真实价格
            self.load_data[self.timestep],            # 真实负荷
            self.ev_data[self.timestep],              # 真实EV需求
            self.soc,                                 # 当前SOC
            self.timestep / len(self.price_data),     # 时间进度
            self.battery_capacity * self.soh          # 有效容量
        ])
        
    def step(self, action):
        if self.timestep >= len(self.price_data):
            return self._get_obs(), 0, True, {}
            
        price = self.price_data[self.timestep]
        cost = 0.0
        energy = 0.0
        
        if action == 1:  # 充电
            energy = self.max_charge_rate * self.dt * self.efficiency
            cost = energy * price / 1000  # $/MWh -> $/kWh
            self.soc = min(1.0, self.soc + energy / (self.battery_capacity * self.soh))
            self.total_cost += cost
            self.total_energy += energy
            
            # SOH轻微衰减
            self.soh = max(0.8, self.soh - 0.000001)
            
        reward = -cost
        self.timestep += 1
        done = self.timestep >= len(self.price_data)
        
        info = {
            'cost': cost, 'energy': energy, 'soc': self.soc, 
            'soh': self.soh, 'price': price, 'action': action
        }
        return self._get_obs(), reward, done, info

def evaluate_strategy(agent, env, strategy_name, episodes=365):
    """评估单个策略"""
    logger.info(f"开始评估 {strategy_name} 策略: {episodes}个回合")
    
    results = []
    for episode in range(episodes):
        obs = env.reset()
        ep_reward = 0.0
        ep_cost = 0.0
        ep_energy = 0.0
        ep_actions = []
        
        done = False
        while not done:
            action = agent.get_action(obs)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_cost += info['cost']
            ep_energy += info['energy']
            ep_actions.append(action)
            
        results.append({
            'episode': episode,
            'strategy': strategy_name,
            'reward': ep_reward,
            'cost': ep_cost,
            'energy': ep_energy,
            'final_soc': env.soc,
            'final_soh': env.soh,
            'charge_rate': np.mean(ep_actions)  # 充电频率
        })
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean([r['reward'] for r in results])
            avg_cost = np.mean([r['cost'] for r in results])
            logger.info(f"{strategy_name}: {episode + 1}/{episodes}, "
                       f"奖励:{avg_reward:.2f}, 成本:{avg_cost:.2f}")
    
    return results

def main():
    """主函数 - 评估5种传统策略"""
    print("=" * 70)
    print("5种传统策略评估")
    print("使用与RL训练相同的环境和数据")
    print("=" * 70)
    
    # 验证数据文件
    required_files = [
        "data_processed/load_15min_fullyear.parquet",
        "data_processed/price_15min_fullyear.parquet", 
        "data_processed/ev_demand_15min_fullyear.parquet",
        "data_processed/soh_params_fullyear.json"
    ]
    
    print("\n验证数据文件...")
    for f in required_files:
        if not Path(f).exists():
            print(f"错误: 缺失 {f}")
            print("错误: 缺少必要数据文件")
            return False
        print(f"已找到: {f}")
    
    # 加载数据
    print("\n加载数据...")
    load_data = pd.read_parquet(required_files[0])
    price_data = pd.read_parquet(required_files[1])
    ev_data = pd.read_parquet(required_files[2])
    with open(required_files[3], 'r') as f:
        soh_params = json.load(f)
    
    print("数据加载完成")
    
    # 创建标准环境
    env = EVVESEnv(load_data, price_data, ev_data, soh_params)
    
    # 定义5种传统策略
    strategies = [
        ("ToU", ToUAgent(), "分时电价策略"),
        ("SafeBot", SafeBotAgent(), "安全机器人策略"), 
        ("MPC", MPCAgent(), "模型预测控制策略"),
        ("Greedy-Price", GreedyPriceAgent(), "贪婪价格策略"),
        ("Random", RandomAgent(), "随机基准策略")
    ]
    
    print(f"\n开始评估5种传统策略...")
    start_time = time.time()
    
    all_results = []
    strategy_stats = {}
    
    # 逐个评估策略
    for strategy_name, agent, description in strategies:
        print(f"\n--- {strategy_name}: {description} ---")
        strategy_results = evaluate_strategy(agent, env, strategy_name, episodes=365)
        all_results.extend(strategy_results)
        
        # 计算统计信息
        rewards = [r['reward'] for r in strategy_results]
        costs = [r['cost'] for r in strategy_results]
        energies = [r['energy'] for r in strategy_results]
        charge_rates = [r['charge_rate'] for r in strategy_results]
        
        strategy_stats[strategy_name] = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_cost': np.mean(costs),
            'std_cost': np.std(costs),
            'mean_energy': np.mean(energies),
            'std_energy': np.std(energies),
            'mean_charge_rate': np.mean(charge_rates),
            'total_cost': np.sum(costs),
            'total_energy': np.sum(energies)
        }
        
        print(f"完成: {strategy_name}, 奖励={strategy_stats[strategy_name]['mean_reward']:.2f}, "
              f"成本={strategy_stats[strategy_name]['mean_cost']:.2f}")
    
    total_time = time.time() - start_time
    
    # 保存结果
    print("\n保存评估结果...")
    Path("frontend_models").mkdir(exist_ok=True)
    
    # 详细结果
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("frontend_models/traditional_strategies_results.csv", index=False)
    
    # 统计摘要
    summary = {
        'evaluation_info': {
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'data_source': '100% Real Australian Electricity Market Data',
            'episodes_per_strategy': 365,
            'total_strategies': len(strategies)
        },
        'strategy_statistics': strategy_stats
    }
    
    with open("frontend_models/traditional_strategies_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 显示最终结果
    print("\n" + "=" * 70)
    print("5种传统策略评估完成")
    print("=" * 70)
    print(f"总评估时间: {total_time:.1f}秒 ({total_time/60:.2f}分钟)")
    print(f"数据来源: 澳洲电力市场数据")
    print()
    
    print("策略性能排名 (按平均奖励):")
    sorted_strategies = sorted(strategy_stats.items(), 
                             key=lambda x: x[1]['mean_reward'], reverse=True)
    
    for i, (name, stats) in enumerate(sorted_strategies, 1):
        print(f"{i}. {name:12} | 奖励: {stats['mean_reward']:6.2f} | "
              f"成本: {stats['mean_cost']:.2f} | 能量: {stats['mean_energy']:6.2f} kWh")
    
    print(f"\n结果文件:")
    print(f"详细结果: frontend_models/traditional_strategies_results.csv")
    print(f"统计摘要: frontend_models/traditional_strategies_summary.json")
    
    print("\n5种传统策略评估完成")
    print("已使用真实数据完成评估")
    print("\n策略对比完成:")
    print("- DDDQN-PER (4种子)")
    print("- Baseline DQN (4种子)")  
    print("- Rule-Based")
    print("- ToU, SafeBot, MPC, Greedy-Price, Random (5种传统策略)")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 