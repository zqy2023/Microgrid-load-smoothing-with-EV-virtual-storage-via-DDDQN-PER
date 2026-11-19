# Single-network DDDQN/PER transfer experiment

这是已完成实验的可移植代码、已核验数据和模型包。本地提交与 GitHub 应使用同一份 `package_manifest.json` 白名单载荷。它提供 4 个教师辅助学生、4 个同种子无教师 TD 对照、独立核验和完整训练入口；不是 2025 年原论文全部结论的原样复现。

## 已完成结果

每组 4 个种子，各 1,200,000 次真实仿真交互；8 次正式运行合计 9,600,000 次交互、2,399,944 次 TD 更新。3 个先行探索实验另计 720,000 次交互，不混入正式预算。训练与评价使用同一个固定的 17,520 行归档窗口，不能解释为新数据泛化。

以下是 4 种子均值，前三项为百分比；完整逐种子值与范围见 `reference/results.json`。

|口径 / 方法|全期方差降幅|日内方差降幅|账单节省|期末减期初电量 kWh|
|---|---:|---:|---:|---:|
|自由期末 / 同期无教师 TD 对照|81.884509|94.032759|8.380653|18.731250|
|自由期末 / 教师辅助单网络|86.880726|95.474542|8.570841|33.731250|
|统一期末 / 同期无教师 TD 对照|77.219710|93.020215|8.541466|0|
|统一期末 / 教师辅助单网络|78.740209|94.265931|8.858808|0|

自由期末执行只使用一个 15 输入、21 输出神经网络的最大 Q 动作。统一期末评价额外应用使用未来可用性信息的命令修正规则，不能称为单网络自己学会闭合电量。每个模型两种口径全部保留，没有选择最佳种子替代四种子结果。

## 环境与安装

实际核验环境：CPython 3.8.0、CPU、PyTorch 2.4.1、NumPy 1.24.4、pandas 2.0.3、pyarrow 17.0.0、Gymnasium 1.0.0、Stable-Baselines3 2.4.1，单线程。建议使用独立虚拟环境，安装 `requirements.txt`。未打包解释器或第三方库。其他平台/版本可运行，但不保证浮点结果逐比特一致；Python 3.8 是历史实验兼容环境，不是生产服务部署建议。

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

以下命令在本包目录执行；也可从任意目录用 `run.py` 的完整路径调用。输出目录必须是新的，不覆盖历史证据。

## 直接推理与独立核验

```sh
python run.py evaluate --run transfer_seed0 --output ../new_eval_seed0
python run.py verify --output ../new_verification
python run.py verify --replay --output ../new_full_replay
```

`verify` 核对所有静态文件哈希，独立重算 8 个模型的初始化、自由期末、统一期末共 24 个完整窗口的物理账本、15 维观测和命名权重矩阵 Q。`--replay` 另用生产推理入口重跑 8 个终点模型的 16 个窗口，再逐数组比较保存轨迹。它们不会重训模型，也不能替代完整训练。

## 重建教师与训练

完整训练保持原已完成实验的预算和方法，不靠缩短步数冒充复现。

```sh
python run.py build-teacher
python run.py smoke --method refined --steps 80032 --seed 0 --output ../new_training_smoke
python run.py audit-training --run ../new_training_smoke --output ../new_training_smoke_audit.json
python run.py train --method refined --seed 0 --output ../new_transfer_seed0
python run.py train --method control --seed 0 --output ../new_control_seed0
python run.py train-matrix --output ../new_full_matrix
```

`train-matrix` 顺序执行 4 个 control 与 4 个 refined，各 120 万步；不会自动降预算。`smoke` 明确是工程短验，默认 80,032 步，会跨过 80k 教师切换点。每 40k 步保存全部真实 transition、观测、动作、奖励和 TD 抽样/损失/教师标签。正式运行还保留 240k/600k/960k 中间网络。最终保存的优化器/目标网络文件不含完整回放优先级、全部随机状态和运行器状态，因此本版本**不声称支持精确中断续训**。

选定的 beta=2 离线教师表从包内数据确定性重建，约 420 MB，放在忽略的 `cache/` 中。生成后必须与历史已审表哈希完全一致；无需联网下载教师表。仅这张已选表会生成，不重新挑选教师或种子。训练还使用包内 10 个冻结教师模型；它们的既有训练成本是复用知识，不能算成当前新增交互。

## 方法和限制

- 真实 21 输出 Dueling 网络，共 36,502 个参数；Double TD、PER、200k 回放容量、batch 32、学习率 3e-4、gamma 0.99；第 32 步起每 4 步更新。目标网络在首次 TD 后以及此后每 1,000 次 TD 同步。Dropout=0，探索前 120k 步线性从 1 降到 0.05。
- 使用同一个树式 PER 的新无教师对照。它不是旧线性采样程序在所有浮点和随机数轨迹上的逐比特替身。
- 21 档总功率为 -1 到 +1 kW、步长 0.1 kW，是总 ±3 kW 上限内的较小动作集合；75 kWh，初始 37.5 kWh，SOC [0,1]，15 分钟步长，保留归档可用性和 SoH。没有扩大容量、强制可用或额外 ramp 约束。
- 15 维观测包含真实未来 5 步价格和 1 步负荷；这是继承的完美前视条件。奖励目标为训练均值负荷加 `(37.5-E_before)/96h` 库存反馈；负平方平滑项与经济项系数 0.02。库存反馈改变奖励，不是给动作添加解析 Q 分数。
- refined 前 80k 步用固定十模型教师，之后持续用读取完整未来窗口的离线 DP 教师；辅助损失权重 4。教师在训练后半程没有关闭。推理入口没有教师依赖，但结果应称为**教师辅助 DDDQN/PER 单网络迁移**，不能称为纯 RL 独立发现，也不能由本实验隔离证明 TD 或 PER 必不可少。
- 没有 teacher-only 对照。同期 control 对比说明在同一 TD 配置下加入教师的收益，不隔离 TD 本身的贡献。
- 全期指标使用全部 17,520 行；日内指标是 182 个完整 96 步日方差均值之比，尾 48 行不纳入日内分母。账单分母是同窗零储能账单，原始计价为价格乘功率乘 0.25/1000。
- 自由期末库存不同，必须同时报告电量变化；统一期末又有额外执行修正，两者不能混称。98.9% 与 44.7% 不是本包已达到的结论；历史 2025 数据溯源和 44.7% 原始比较基线仍未确认。

## 证据与分发边界

`reference/expected.json` 与原结果逐项绑定，`reference/source_provenance.json` 记录原源 SHA 与便携版本差异；`reference/portable_validation.json` 记录本包实际执行的验证范围。历史完整训练块未放进公开包，原完整训练审计只保留来源哈希和明确范围；新运行会生成可独立核查的训练块。审计不声称重演全部历史梯度或未保存的教师软 logits。

只发布 `package_manifest.json` 列出的静态文件及该清单本身。缓存、虚拟环境、新运行输出、个人 CV 和原提交目录都不在发布载荷中。数据和代码权利说明见 `DATA_NOTICE.md`；未添加未经证实的许可证。
