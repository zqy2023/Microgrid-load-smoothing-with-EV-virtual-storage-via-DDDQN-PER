# Benchmark results and scope

## Main archive

Results use the 17,520-point archive, shared physical dispatch constraints and signed billing. The within-day denominator uses the 182 complete days. The full-window and billing metrics include the final 48 intervals. Savings and variance reduction use zero storage as their reference. A negative reduction means that variance increased. ToU and MPC run one deterministic policy in each terminal convention. The DDDQN/PER and DDPG rows are means over four seeds.

| Terminal rule | Method | Whole-window variance reduction | Within-day variance reduction | Bill saving | Final minus initial energy |
| --- | --- | ---: | ---: | ---: | ---: |
| Equal | DDDQN/PER | 78.74% | 94.27% | 8.86% | 0.00 kWh |
| Equal | ToU | −78.44% | −92.84% | 17.84% | 0.00 kWh |
| Equal | MPC | 78.73% | 94.79% | 8.70% | 0.00 kWh |
| Equal | DDPG | 78.12% | 94.00% | 8.56% | 0.00 kWh |
| Free | DDDQN/PER | 86.88% | 95.47% | 8.57% | +33.73 kWh |
| Free | ToU | −76.75% | −92.14% | 18.26% | −34.00 kWh |
| Free | MPC | 83.32% | 95.81% | 8.54% | +18.22 kWh |
| Free | DDPG | 82.54% | 94.90% | 8.42% | +17.41 kWh |

In the equal-terminal comparison, the DDDQN/PER mean exceeds the DDPG mean by 0.62, 0.27 and 0.29 percentage points on the three metrics. DDDQN/PER has higher bill savings in all four corresponding seed comparisons and higher smoothing in two of four for each variance metric. Relative to MPC, DDDQN/PER gains about 0.01 percentage points on whole-window reduction and 0.16 on bill saving. MPC gains about 0.52 percentage points on within-day reduction. The DDDQN/PER peak-to-valley gap grows by 20.20% in the equal-terminal window.

## Date and availability sensitivity

The later archive runs continuously from 3 July through 31 December 2023. The load series repeats every 17,520 intervals. Price and EV availability provide the changed archived inputs. In that window under equal terminal energy, DDDQN/PER obtains 78.72%, 94.39% and 8.80%. DDPG obtains 79.12%, 95.21% and 8.92%. MPC obtains 79.70%, 95.93% and 8.79%. The sectioned seasonal runs also depend on their starting inventories and shorter evaluation horizons.

The CSIRO-based scenario multipliers act on the archived EV demand used for the availability gate. The half-demand stress case reduces DDDQN/PER's whole-window variance reduction from 78.72% to 59.48% and bill saving from 8.80% to 6.60% under equal terminal energy. The gate is a simulation proxy for aggregate availability.

A separate offline sensitivity uses average load from a different 300-household SGSC cohort and contemporaneous AEMO prices. The original half-hour averages are held across two 15-minute simulation intervals. EV availability is carried over from the archive for that exercise. Under equal terminal energy, DDDQN/PER increases full-window variance in all four season windows. The repository provides the derived seasonal metrics and an optional replay for users who have the expected aggregate inputs.

## Verification and interpretation

The prior CPU checks include complete DDDQN/PER retraining and replay, and four full 1.2-million-interaction DDPG runs. The repository verification commands repeat inference, physical ledger and metric checks with the saved weights. The DDDQN/PER training and primary evaluation share the same archive. Auxiliary offline targets use that full window. The separate terminal rule uses archived future EV availability. The available evidence supports comparison under these simulation protocols. Matched native five-minute household, price and EV inputs and Jetson Orin NX latency tests remain unverified.

- [Main comparison values](benchmark/reference/primary_comparison.tsv)
- [Archive scenario values](benchmark/reference/archive_scenarios.tsv)
- [External seasonal sensitivity](benchmark/reference/external_seasons.tsv)
- [Commands and source bindings](benchmark/README.md)
- [Frozen DDDQN/PER training and verification](reconstruction/README.md)
