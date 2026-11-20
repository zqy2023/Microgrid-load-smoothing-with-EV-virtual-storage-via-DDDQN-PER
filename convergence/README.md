# Sampled convergence and training time

Eight training runs compare the proposed 21-action DDDQN/PER policy with the same network trained using uniform replay with replacement. Four paired random seeds each complete 1,200,000 interactions per arm. The runs share archive inputs, reward, two-stage auxiliary action targets, network initialization within each pair and the evaluation protocol. The main evaluation window contains 17,520 15-minute intervals and uses equal initial and terminal storage energy of 37.5 kWh.

The primary target is a 75% reduction in whole-window variance relative to zero storage. A sustained crossing requires every later sampled checkpoint to meet the target, with at least 40,000 further interactions observed. The protocol also reports 70% and 77% thresholds. Crossings are observed at the sampled checkpoints.

| Four-seed average | DDDQN/PER | DDDQN with uniform replay |
| --- | ---: | ---: |
| First checkpoint meeting 75% | 25,000 interactions | 27,500 interactions |
| Sustained crossing of 75% | 576,500 interactions | 560,000 interactions |
| Process CPU time for 1,200,000 interactions | 2,184.42 seconds | 2,075.39 seconds |

At the first 75% crossing, PER reaches the target earlier in three of four pairs. At the sustained crossing, PER also leads in three of four pairs and reaches the target later on average because one seed takes 600,000 more interactions. At the fixed budget, the mean of the paired relative times shows a 5.25% higher process CPU cost for PER. The timing applies to the two implementations and the recorded environment. The test measures in-window performance under the shared simulator protocol. The 2025 historical Base DQN used a separate configuration.

- [Frozen protocol](protocol.json)
- [Chinese verification report](results/N13收敛速度核验报告.md)
- [Checkpoint curves](results/checkpoint_convergence.pdf)
- [Per-seed crossings and timing](results/per_seed.tsv)
- [All 672 checkpoint observations](results/checkpoint_curve.tsv)
- [Machine-readable paired summary](results/convergence_summary.json)
- [Training-journal audit receipt](results/training_journal_audit.json)
- [Recorded training environment](reference/environment.json)

## Replay from a clean checkout

Run from the repository root using the Python 3.8 CPU requirements. Use a fresh output directory. The reference run folders include the exact 84 saved checkpoints and formal receipt for each run.

~~~sh
python -m pip install -r requirements.txt
python convergence/verify.py --output convergence_check
~~~

The command replays all 672 checkpoints and verifies the receipt bindings, action and physical trace hashes, simulated metrics and paired summaries. The replay performs zero new training interactions. CPU time is read from the original training records. The clean replay uses the published inputs and static model weights.

## Repeat training and audit

Build the fixed dynamic-programming target table and provide fresh directories for the new runs and offline evaluations. Use the same one-thread-per-run budget and four paired seeds. Process CPU times will vary by host.

~~~sh
python reconstruction/run.py build-teacher
mkdir fresh_runs
mkdir fresh_evaluations
python convergence/run_cohort.py --repo . --runs fresh_runs --evaluations fresh_evaluations --workers 8
python convergence/summarise.py --evaluations fresh_evaluations --output fresh_results
python convergence/audit_journals.py --runs fresh_runs --output fresh_journal_audit.json
~~~

The retraining generates the raw transition and optimizer journals used by the independent audit. The original full journals are retained in the private archive. The GitHub release contains the per-run receipts, static weights, offline evaluations and journal audit result.

To render the curve again from either table, install Matplotlib 3.7.5 and create a fresh output folder.

~~~sh
mkdir fresh_figures
python convergence/plot_curve.py --curve convergence/results/checkpoint_curve.tsv --output-dir fresh_figures
~~~
