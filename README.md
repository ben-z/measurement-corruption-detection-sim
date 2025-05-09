# measurement-corruption-detection-sim

This repository contains the simulation code for the thesis "Optimal Sensor Protection and Measurement Corruption Detection in Safety-Critical Systems". It has multiple versions of kinematic bicycle simulators with sensor fault injection and detection analysis.

## Repository Structure

- **bsim/** (Version 1)
  - `Dockerfile`
  - `install-dependencies.sh`: setup script for backend and frontend
  - **backend/**: Python backend service
  - **frontend/**: Web-based frontend
- **bsim_v2/** (Version 2 - experimental)
  - `main.py`, `run_exp.py`: simulation and batch experiment scripts
  - **analysis/**: analysis utilities
  - `fault_generators.py`, `utils.py`: fault injection and helper functions
  - Tests: `test_utils.py`, `test_fault_generators.py`
- **bsim_v3/** (Version 3 - current)
  - `main.py`: single-run simulation script
  - `run_sim.py`: CLI for batch simulations (Typer)
  - `run_sim_debug.py`: debugging runner
  - `slurm_sweep.sh`: SLURM job script for parameter sweeps
  - `analysis.py`: Marimo-powered analysis cells
  - **lib/**: modular library of controllers, detectors, estimators, planners, plants, sensors
  - `requirements.txt`

## Version 1 (bsim)

### Overview
Version 1 was the initial release: a Python backend simulation engine with sensor fault injection and detection, paired with a web-based visualizer for real-time plotting.

Key features:
- Python backend: continuous and discrete kinematic bicycle models, planners, controllers, and detectors
- Web frontend: live plots via WebSocket and uPlot

### Prerequisites
- Docker (optional) or Conda & Node.js

### Setup and Usage
```bash
cd bsim
./install-dependencies.sh
# Start the backend
cd backend
conda activate bsim-backend
python main.py
# Serve the frontend (open index.html or run webpack dev server)
cd ../frontend
npm run dev
```
Refer to `bsim/backend/README.md` and `bsim/frontend/README.md` for more details.

## Version 2 (bsim_v2)

### Overview
Version 2 migrated to a pure-Python codebase, consolidating simulation, fault generation, detection, and analysis into a single environment. This release focused on significant performance improvements in the fault detector.

Changelog / improvements:
- Removed separate web frontend; analysis via scripts and Jupyter
- Optimized detector algorithms with CVXPY and vectorized operations for speed
- Added batch experiment scripts (`run_exp.py`) and table generation utilities

### Prerequisites
- Python 3.8+ and pip

### Setup and Usage
```bash
cd bsim_v2
pip install -r requirements.txt
# Run simulations or experiments
python main.py
python run_exp.py
```

## Version 3 (bsim_v3)

### Overview
Version 3 refined and simplified V2, emphasizing modularity, code quality, and further speed gains. Core components were moved into a reusable `lib/` folder, and a CLI (Typer) plus Marimo notebook support were introduced. This version powered the accompanying thesis work.

Changelog / improvements:
- Modular library: controllers, detectors, estimators, planners, plants, sensors under `lib/`
- CLI interface for batch simulations (`run_sim.py`) and SLURM sweep support
- Performance optimizations: streamlined simulation loops, reduced dependencies
- Improved maintainability: type annotations, clearer structure, interactive analysis via Marimo

### Prerequisites
- Python 3.8+ and pip
- (Optional) Marimo for interactive analysis: `pip install marimo`

### Setup
```bash
cd bsim_v3
pip install -r requirements.txt
```

### Running Simulations
- Single run:
  ```bash
  python main.py
  ```
- Batch simulations:
  ```bash
  python run_sim.py run-multiple --num-simulations N \
      --out-file-template "exp/bsim_v3/results-<id>.parquet" \
      [--fault-type TYPE] [--eps-scaler VALUE] [--force-detection]
  ```
- Debug mode:
  ```bash
  python run_sim_debug.py [args]
  ```
- SLURM parameter sweep:
  ```bash
  sbatch slurm_sweep.sh
  ```

### Analysis
Launch the Marimo notebook to explore results and generate visualizations:
```bash
pip install marimo
marimo notebook analysis.py
```
Processed data and plots are saved under `visualizations/data/bsim_v3/`.

## Contributing
Contributions, issues, and feature requests are welcome. Please open an issue or submit a pull request.


## Related Links
- Deployment code for a Clearpath Jackal robot: https://github.com/ben-z/measurement-corruption-detection-jackal
