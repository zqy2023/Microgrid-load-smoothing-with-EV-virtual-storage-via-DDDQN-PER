# Validation summary

The 2026 archived-input study completed eight formal training runs with 1,200,000 interactions and 299,993 TD updates each. The four transfer runs and their four teacher-free controls shared seeds, initial weights, physical settings and training budget. An additional 720,000 exploration interactions were recorded separately.

Independent checks covered 10,320,000 total training transitions, 82,557,536 replay sample links and all 33 research evaluation windows, including initial weights. Both the load and bill ledger were recalculated on actual executed power. Saved final models use a single 15-input, 21-output network. A separate inference entry point reproduced one formal model's two full-window traces exactly.

The primary study results are the fixed 1,200,000-step endpoints. The 86.88% full-window and 95.47% intraday reductions and 8.57% bill savings describe the archived simulation and idle-storage baseline. They do not replace the historical measurements or certify the initial theses. The matched terminal-energy result is a separately applied execution rule. The archived window is shared across all seeds.

For per-seed figures, archived input hashes, checked model weights and scope of the included public evidence, see the [reproduction package](reconstruction/).
