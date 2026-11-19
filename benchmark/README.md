# Four-controller comparison

This directory reproduces the benchmark used in the revised Chinese and English theses. It compares the released DDDQN/PER policy, ToU, MPC and DDPG, with zero storage as the common reference. The separate sealed `reconstruction/` directory and all of its original audit arms retain their original bytes and identifiers.

## Replay from this checkout

Install the root `requirements.txt` in a dedicated Python 3.8 CPU environment. Run the commands from the repository root.

```text
python benchmark/run.py verify --output ../primary-verification --jobs 4
python benchmark/run.py verify --scope archive --output ../archive-verification --jobs 4
```

The primary replay checks 20 full windows and 350,400 simulated 15-minute steps. The archive replay checks 200 windows and 2,623,680 steps, including primary. Both check file hashes, per-step command index hashes, battery and billing constraints, and metrics against the audited reference. The primary DDDQN/PER trajectories are also compared with the original sealed arrays. Each run writes `verification.json` and `comparison.tsv`. Use `--save-traces` to keep the regenerated arrays. The output must be a new directory. Running with one job uses less memory.

Replay uses the frozen policies and performs zero training interactions. The released DDDQN/PER is the model stored under `reconstruction/models/transfer_seed*`; the legacy directory name is maintained for byte-exact auditing. The four DDPG model archives are hash-checked before loading. [The training record](reference/ddpg_training.json) includes four completed 1.2-million-interaction runs and the resulting model hashes.

## Train locally

```text
python reconstruction/run.py build-teacher
python reconstruction/run.py train --method refined --seed 0 --output ../dddqn-seed0
python benchmark/baseline_ddpg/run_ddpg.py --archive reconstruction --seed 0 --steps 1200000 --output ../ddpg-seed0
python benchmark/baselines_tou_mpc/run.py --package reconstruction --output ../tou-mpc
```

Repeat neural training for seeds 0–3. The historical CLI token `refined` selects the published DDDQN/PER training configuration. It combines the TD/PER objective with auxiliary action targets from ten frozen members for the first 80,000 steps, then uses an offline dynamic program with access to the whole archive window. The frozen members' original training and building that table do not count toward the reported interaction budgets. At inference, the published DDDQN/PER uses one network. DDPG uses a continuous proposal rounded to the same 21-action execution grid.

## Comparison

The primary window spans 17,520 archive rows starting 2023-01-01. The effective command grid is ±1 kW in 0.1 kW increments, with 75 kWh capacity, 37.5 kWh initial inventory, simulated availability and signed billing. Every active method has a free-terminal run and a return-to-initial-energy run. The equal-terminal feasibility wrapper uses future availability. The ToU schedule is calibrated in-window and MPC replans with a six-step horizon and archived forecast inputs. The methods do not have identical policy information sets.

| Method, equal terminal energy | Full-window variance reduction | Within-day variance reduction | Bill savings |
|---|---:|---:|---:|
| DDDQN/PER | 78.74% | 94.27% | 8.86% |
| ToU | −78.44% | −92.84% | 17.84% |
| MPC | 78.73% | 94.79% | 8.70% |
| DDPG | 78.12% | 94.00% | 8.56% |
| Zero storage | 0.00% | 0.00% | 0.00% |

The table reports the mean across four seeds for DDDQN/PER and DDPG. ToU and MPC are deterministic policies with one realization per terminal protocol. DDDQN/PER had slightly higher means than DDPG on the primary window. MPC had better within-day smoothing and ToU had greater bill savings while increasing variance. DDDQN/PER's equal-terminal peak-to-valley range increased by 20.20%. In the archived July–December scenario, DDPG had higher means on all three reported metrics and MPC smoothed more.

- [Primary results, both terminal protocols](reference/primary_comparison.tsv)
- [All ten archive scenarios](reference/archive_scenarios.tsv)
- [Derived EV availability sensitivity multipliers](reference/ev_scenarios.json)
- [Separate external four-season results](reference/external_seasons.tsv)

Training and the primary comparison use the same archive window. The later date windows use a repeated load series, and the EV multipliers adjust the archive availability proxy. These are sensitivity tests. They do not demonstrate validation on independent household loads, actual penetration rates, field performance, or hardware.

## Optional external four-season check

The separate sensitivity study uses 300 household historical load records (SGSC) aligned to AEMO price, with the archived EV availability proxy. It evaluates the 2013–2014 four-season windows for the published DDDQN/PER models. The table above is reproducible from repository inputs. The external input is supplied by the operator and is not distributed here.

```text
python benchmark/run.py external --input PATH_TO_ALIGNED_CSV --output ../external-verification --jobs 4
```

The expected tabular columns are `timestamp`, `load_kw` and `price_aud_mwh`, with 40,704 regularly spaced 15-minute rows starting 2013-01-01. Incoming half-hour load and price values in this archived aligned table occupy two adjacent rows. The file SHA-256 must equal `2037f235cd3e02b041a7ae7f17283d99a60597e9b14e3b34559f98e1ffa91ff8`. The optional replay checks 32 windows and 279,552 simulated steps, including all trace-array digests. These results remain a separate sensitivity experiment in the comparison.

