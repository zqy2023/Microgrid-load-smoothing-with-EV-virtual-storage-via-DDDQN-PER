# EV virtual storage load smoothing

This repository's current experiment is a 2026 reconstruction on the processed inputs preserved in the undergraduate submission archive. It compares a teacher-assisted single 21-action Dueling Double DQN with prioritized replay (DDDQN/PER) to the same learner trained without teacher guidance. See the [reproduction package](reconstruction/) for the training, evaluation and verification commands.

The historical NEU thesis and UTS Capstone results are a separate record. This study does not authenticate the historical raw data, recover its original checkpoints, or verify the 98.9% variance-reduction and 44.7% cost-saving claims.

## Measured results

Each arm used four matched seeds and 1,200,000 interactions per seed. The processed simulation used the same 17,520 archived rows, a 75 kWh equivalent storage unit, and 21 fixed action commands from −1 to +1 kW at 0.1 kW intervals.

| Free terminal inventory, same window | Full-window variance reduction | Intraday variance reduction | Modeled bill savings |
|---|---:|---:|---:|
| Unassisted DDDQN/PER, four-seed mean | 81.88% | 94.03% | 8.38% |
| Historical ten-network ensemble reference | 82.96% | 95.27% | 8.48% |
| Teacher-assisted single network, four-seed mean | 86.88% | 95.47% | 8.57% |
| Matched single-network improvement | +5.00 pp | +1.44 pp | +0.19 pp |

All four students exceeded the ensemble on all three free-terminal metrics. Mean terminal energy was 71.23 kWh for the students and 56.23 kWh for the controls. In a separate assessment with both groups returned to 37.5 kWh by the same execution rule, the students achieved 78.74%, 94.27% and 8.86%. The matched improvements in that assessment were 1.52, 1.25 and 0.32 percentage points. Bill savings use the same-window idle-storage bill as denominator. The terminal execution rule consults archived future availability.

Training uses an existing ensemble and a full-window offline schedule as teaching signals. Inference selects an action from a single network. The policy was trained and evaluated on the same archived window. This comparison measures teaching added to DDDQN/PER on that window; it does not isolate a PER advantage, establish generalization, or test a physical vehicle fleet.

## Project materials

- [Run instructions and validation scope](reconstruction/README.md)
- [Result summary](REPORT.md)
- [Historical main snapshot](https://github.com/zqy2023/Microgrid-load-smoothing-with-EV-virtual-storage-via-DDDQN-PER/tree/522396cc3f068935fd9e0b7ce33e458b943157de)
- [Historical distributional DQN branch](https://github.com/zqy2023/Microgrid-load-smoothing-with-EV-virtual-storage-via-DDDQN-PER/tree/feature/distributional-dqn)

当前主分支以 2026 年归档数据重建为复现实验入口。中英文 2025 原稿、早期代码和 QR-DQN 支线为历史资料。复跑、模型和核验请从上面的新包入口开始。
