# Microgrid Load Smoothing with EV Virtual Storage via DDDQN-PER

This repository implements **Dueling Double Deep Q-Network with Prioritized Experience Replay (DDDQN-PER)** for microgrid load smoothing using Electric Vehicle (EV) virtual storage.

## 📋 Overview

The project uses reinforcement learning to optimize EV charging/discharging schedules for:
- **Economic optimization**: Minimizing electricity costs
- **Battery health preservation**: Reducing degradation
- **Grid stability**: Smoothing load fluctuations

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher (tested on Python 3.12.3)
- pip package manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/zqy2023/Microgrid-load-smoothing-with-EV-virtual-storage-via-DDDQN-PER
cd Microgrid-load-smoothing-with-EV-virtual-storage-via-DDDQN-PER

# 2. Upgrade pip and install dependencies
python3 -m pip install -U pip setuptools wheel
pip install -r requirements.txt

# 3. Verify installation
python3 tests/test_smoke.py
```

Expected output:
```
✓ test_standard_library_imports
✓ test_project_imports
✓ test_evves_env_class
✓ test_prioritized_replay_buffer
✓ test_gymnasium_compatibility
✓ test_stable_baselines3_dqn
✓ test_torch_available

All 7 tests passed!
```

## 📦 Dependencies

Core dependencies (automatically installed):
- **NumPy** >=1.24.0 - Numerical computing
- **Pandas** >=2.0.0 - Data manipulation
- **PyTorch** >=2.0.0 - Deep learning framework
- **Gymnasium** >=0.28.0 - RL environment interface
- **Stable-Baselines3** >=2.0.0 - RL algorithms
- **Matplotlib** >=3.7.0 - Visualization
- **Seaborn** >=0.12.0 - Statistical visualization
- **PyArrow** >=12.0.0 - Data serialization

See `requirements.txt` for complete list.

## 🏗️ Project Structure

```
.
├── requirements.txt              # Python dependencies
├── REPORT.md                    # Detailed analysis report
├── tests/                       # Test suite
│   └── test_smoke.py           # Import and basic functionality tests
└── 4-程序及其他相关材料/
    ├── 01_source_code/
    │   └── code/               # Main source code
    │       ├── evves_env.py    # RL environment implementation
    │       ├── dddqn_per_sb3.py # DDDQN-PER algorithm
    │       ├── train_dddqn.py  # Training script
    │       ├── baseline_dqn_train.py  # Baseline DQN
    │       ├── train_dddqn_paper.py   # Paper config training
    │       └── utils/          # Utility functions
    ├── 02_datasets/            # Data files
    ├── 03_analysis_scripts/    # Analysis tools
    ├── 04_Drawing program/     # Visualization scripts
    └── 05_figure/             # Generated figures
```

## 🎯 Usage

### Training

The project provides multiple training scripts:

#### 1. DDDQN-PER Training (Recommended)

```bash
cd "4-程序及其他相关材料/01_source_code/code"

python3 train_dddqn.py \
  --load <path_to_load.parquet> \
  --price <path_to_price.parquet> \
  --ev <path_to_ev.parquet> \
  --soh <path_to_soh_params.json> \
  --steps 100000 \
  --save_dir ./models
```

#### 2. Baseline DQN Training

```bash
python3 baseline_dqn_train.py \
  --load <path_to_load.parquet> \
  --price <path_to_price.parquet> \
  --ev <path_to_ev.parquet> \
  --soh <path_to_soh_params.json> \
  --steps 100000
```

#### 3. Paper Configuration Training

```bash
python3 train_dddqn_paper.py \
  --load <path_to_load.parquet> \
  --price <path_to_price.parquet> \
  --ev <path_to_ev.parquet> \
  --soh <path_to_soh_params.json>
```

### Command Line Options

Common options across training scripts:

- `--load`: Path to load demand data (Parquet format)
- `--price`: Path to electricity price data (Parquet format)
- `--ev`: Path to EV availability/demand data (Parquet format)
- `--soh`: Path to battery State-of-Health parameters (JSON format)
- `--seed`: Random seed for reproducibility (default: 0)
- `--steps`: Number of training steps (default: varies by script)
- `--save_dir`: Directory to save trained models
- `--device`: Computing device (cpu/cuda/auto)

## 📊 Data Requirements

Training requires four data files:

1. **Load Data** (`*.parquet`): Time-series grid load demand
2. **Price Data** (`*.parquet`): Electricity pricing information
3. **EV Data** (`*.parquet`): EV availability and charging requirements
4. **SOH Parameters** (`*.json`): Battery health model parameters

Raw data is available in `4-程序及其他相关材料/02_datasets/data/` but requires preprocessing to convert to the required formats.

## 🔍 Environment Details

### State Space (11 dimensions)
- **Price features** (5): Current price, trends, forecasts
- **Load features** (3): Current load, patterns, forecasts  
- **Battery features** (2): State of Charge (SOC), availability
- **Time feature** (1): Time period index

### Action Space (21 discrete actions)
- Range: -50 kW to +50 kW (5 kW steps)
- Negative: Discharge (V2G - Vehicle-to-Grid)
- Zero: Idle
- Positive: Charge

### Reward Function
Multi-objective optimization balancing:
- Economic cost reduction
- Battery degradation minimization
- Grid load smoothing

## 🧪 Testing

Run the smoke tests to verify installation:

```bash
python3 tests/test_smoke.py
```

The test suite validates:
- All required libraries are installed
- Project modules import correctly
- Core classes are accessible
- Basic functionality works

## 📈 Visualization

Visualization scripts are available in `4-程序及其他相关材料/04_Drawing program/`:

- `create_figure_2_1.py`: Load curve visualization
- `figure3_1_load_price_correlation.py`: Price-load correlation
- `figure4_1_learning_monitoring.py`: Training progress monitoring
- `figure5_2_training_convergence_curve.py`: Convergence analysis
- And more...

## 📚 Documentation

- **REPORT.md**: Comprehensive analysis of dependencies and runtime issues
- **COPILOT_INSTRUCTIONS.md**: Development guidelines and best practices
- **copilot-setup-steps.yaml**: Automated setup steps

## 🛠️ Development

### Code Quality

The project follows:
- PEP8 style guidelines
- Google-style docstrings
- Explicit imports (no wildcards)
- Type hints where applicable

### Contributing

When contributing:
1. Run tests before committing
2. Follow existing code style
3. Update documentation as needed
4. Ensure compatibility with Python 3.9+

## ⚠️ Known Issues

1. **Data Preprocessing Required**: Raw data needs to be converted to Parquet format before training
2. **No Sample Data**: Integration testing requires preparing actual data files

See `REPORT.md` for detailed analysis and recommendations.

## 📝 Citation

If you use this code in your research, please cite the original paper (citation details to be added).

## 📄 License

[License information to be added]

## 🤝 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Contact: [contact information to be added]

## 🎓 Acknowledgments

This project implements research on microgrid load smoothing using reinforcement learning and electric vehicle virtual storage systems.

---

**Last Updated**: October 2025  
**Status**: ✅ Runnable - All dependencies working, core functionality tested
