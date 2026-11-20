# EV virtual storage load smoothing

This repository supports the current Chinese and English capstone manuscripts. It compares the proposed single-network DDDQN/PER controller with time-of-use scheduling (ToU), six-step model predictive control (MPC) and grid-quantised deep deterministic policy gradient (DDPG). Zero storage supplies the common load variance and billing reference.

## Primary comparison

The main window contains 17,520 archived 15-minute intervals from 1 January 2023 through 2 July 2023 at 11:45. All methods use the same 75 kWh equivalent storage, 37.5 kWh starting energy, availability gate and signed electricity bill. The 21 candidate power commands span −1 to +1 kW in 0.1 kW increments. The table uses the same terminal rule for all dispatch methods, returning storage to 37.5 kWh with archived future availability.

| Method | Whole-window variance reduction | Within-day variance reduction | Bill saving |
| --- | ---: | ---: | ---: |
| DDDQN/PER | 78.74% | 94.27% | 8.86% |
| ToU | −78.44% | −92.84% | 17.84% |
| MPC | 78.73% | 94.79% | 8.70% |
| DDPG | 78.12% | 94.00% | 8.56% |
| Zero storage | 0.00% | 0.00% | 0.00% |

Each learning method uses four seeds at 1.2 million interactions per seed. DDDQN/PER combines temporal-difference learning with auxiliary action targets from ten frozen models over the first 80,000 interactions and a full-window offline dynamic program thereafter. At inference, one learned network chooses among the 21 candidate commands. DDPG outputs a continuous proposal, which the environment maps to the same candidate grid. MPC uses a short recorded lookahead. ToU uses fixed hours calibrated from the first 28 archived days.

The proposed method has a small mean advantage over DDPG in this window. Its pooled variance reduction is close to MPC's, and MPC achieves better within-day smoothing. ToU saves more on the bill and increases both variance measures. The proposed method's peak-to-valley gap increases by 20.20% under the equal-terminal rule. The later archive and the independent load and price sensitivity yield different rankings. [Read the measured results and scope](REPORT.md).

## Reproduce the results

~~~sh
python -m pip install -r requirements.txt
python benchmark/run.py verify --output comparison_check
python benchmark/run.py verify --scope archive --output archive_check
~~~

The first verification command replays the main comparison from repository assets. The second replays the later archive, seasonal windows and availability sensitivity. Both use saved model weights and add zero training interactions. Consult [benchmark instructions](benchmark/README.md) for the separate training commands and the optional external-data check.

[The core reconstruction package](reconstruction/README.md) contains the original training code, archived inputs, model weights and frozen audit evidence. Paths and method labels in that archive can differ from the current manuscript labels. The current DDDQN/PER checkpoints use the stored transfer configuration.

## Component ablation

Twelve additional full-budget runs remove Double DQN target selection, the Dueling head or PER one at a time. Each component study holds the auxiliary training targets, seeds, data, constraints and evaluation protocol fixed. [Results and saved-model replay](ablation/README.md) include both terminal regimes.

## Sampled convergence and training time

A four-seed paired study compares PER with uniform replay while holding the 21-action DDDQN and training conditions fixed. At the 75% whole-window variance-reduction target, PER reaches the target 2,500 interactions earlier on average at the first measured crossing. The mean sustained crossing is 16,500 interactions later, and PER uses 5.25% more process CPU time over the fixed training budget. [Experimental protocol, complete checkpoint replay and training commands](convergence/README.md).

## 中文说明

本仓库与新版中英文论文使用同一实验口径。正文主比较为本文 DDDQN/PER、ToU、MPC、DDPG 和无储能基线。表中采用统一期末电量 37.5 kWh。可用上面的入口从仓库文件重放，并在结果报告查看自由期末、归档后半年及外部四季的适用边界。

[N13 收敛速度核验报告](convergence/results/N13收敛速度核验报告.md)提供配对种子的首次达标、持续达标和训练计时结果。
