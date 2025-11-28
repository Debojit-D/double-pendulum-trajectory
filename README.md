# Double Pendulum Trajectory Modeling with Neural Networks

This repository contains a comprehensive framework for data collection, model training, and evaluation of neural network architectures for modeling double-pendulum dynamics. The project implements multiple machine learning approaches including Lagrangian Neural Networks (LNN), Hamiltonian Neural Networks (HNN), and SINDy-based methods.

## Project Overview

The primary goal of this project is to develop machine learning models that can accurately predict double-pendulum trajectories under different dynamical regimes:
- **Ideal regime**: No damping or friction
- **Viscous regime**: Damping-based energy dissipation
- **Stiction regime**: Friction loss-based energy dissipation

## Repository Structure

```
├── data/                          # Trajectory data organized by regimes
│   ├── SampleIdeal1/             # Ideal regime datasets
│   ├── SampleIdeal2/             # Ideal regime datasets
│   └── SampleStiction1/          # Stiction regime datasets
│
├── description/                   # MuJoCo simulation model files
│   ├── double_pendulum.xml       # Standard double pendulum MJCF model
│   └── double_pendulum_3regimes.xml
│
├── scripts/                       # Utility and analysis scripts
│   ├── collect_data_with_acceleratations.py     # Generate trajectory data with accelerations
│   ├── collect_data_without_accelerations.py    # Generate trajectory data
│   ├── load_*.py                 # Data loading utilities
│   ├── plot*.py                  # Visualization scripts
│   ├── model_test_val_plot.py    # Training metrics visualization
│   └── plots/                    # Generated plots directory
│
├── src/                          # Core source code
│   ├── data_collection/          # Data collection modules
│   ├── training/                 # Model training implementations
│   │   ├── train_lnn_*.py        # Lagrangian Neural Network trainers
│   │   ├── train_hnn_*.py        # Hamiltonian Neural Network trainers
│   │   ├── train_sindy.py        # SINDy model training
│   │   └── train_pysindy.py      # PySINDy model training
│   └── model_evaluation/         # Evaluation and testing modules
│
└── utils/                        # Utility functions
    └── model_training/           # Training utilities
```

## Key Features

### Data Collection
- **Physics-based simulation** using MuJoCo for accurate trajectory generation
- **Multiple dynamical regimes** for diverse training scenarios
- **CSV export** with state variables, positions, and accelerations
- **Metadata tracking** via JSON manifests

### Model Architectures
1. **Lagrangian Neural Networks (LNN)**: Encodes Lagrangian mechanics into neural network structure
2. **Hamiltonian Neural Networks (HNN)**: Preserves energy conservation through network design
3. **SINDy**: Discovers sparse symbolic dynamics from data
4. **PySINDy**: Python implementation of sparse identification of nonlinear dynamics

### Training Features
- Configurable data splits and validation strategies
- Loss metric tracking and visualization
- Model serialization and checkpointing
- Publication-ready visualization (seaborn-styled plots)

## Quick Start

### 1. Data Collection

Generate trajectory data with accelerations:
```bash
python scripts/collect_data_with_acceleratations.py
```

Generate trajectory data without accelerations:
```bash
python scripts/collect_data_without_accelerations.py
```

### 2. Model Training

Train an LNN model with data-derived accelerations:
```bash
python src/training/train_lnn_data_accV1.py
```

Train an HNN model:
```bash
python src/training/train_hnn_V1.py
```

Train a SINDy model:
```bash
python src/training/train_sindy.py
```

### 3. Evaluation and Visualization

Generate test/validation plots:
```bash
python scripts/model_test_val_plot.py
```

## Data Format

### CSV Trajectory Files
Columns in generated CSV files:
- `t`: Time
- `q1`, `q2`: Joint angles (radians)
- `dq1`, `dq2`: Joint velocities
- `tip_x`, `tip_y`, `tip_z`: End-effector position
- `tip_x_rel`, `tip_z_rel`: Relative end-effector position
- `elbow_x`, `elbow_z`: Elbow joint position
- `step_idx`: Simulation step index
- `ddq1`, `ddq2`: Joint accelerations (when available)

### Metadata (JSON)
Each dataset includes a manifest JSON file with simulation parameters:
- Regime information
- Physical parameters (damping, friction loss)
- Initial conditions
- Simulation time and step count

## Model Evaluation

The training pipeline generates:
- **metrics.json**: Training and validation loss history
- **Model checkpoints**: Serialized model weights
- **Plots**: Training curves and prediction comparisons

## Requirements

Key dependencies:
- Python 3.7+
- MuJoCo (for physics simulation)
- PyTorch (for neural network models)
- NumPy, Pandas (data processing)
- Matplotlib, Seaborn (visualization)
- scikit-learn (optional, for some utilities)

## Usage Notes

- **Regime selection**: Set `REGIME` parameter in data collection scripts (1=IDEAL, 2=VISCOUS, 3=STICTION)
- **Output paths**: Update `OUT_DIR` paths to match your data directory structure
- **XML model**: Ensure MuJoCo XML files are properly configured before running simulations
- **GPU acceleration**: Most training scripts support GPU training when available

## References

This project implements concepts from:
- Lutter et al. (2019): "Lagrangian Neural Networks"
- Cranmer et al. (2020): "Discovering Hamiltonian Mechanics with Unsupervised Learning"
- Brunton et al.: "Discovering governing equations from data"

## Author

Debojit-D

## Course

ME691 XVII: Scientific Machine Learning for Thermo-Fluids
