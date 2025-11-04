## QStockPrediction — README

This directory contains a small project focused on volatility prediction and portfolio PFE (Potential Future Exposure) estimation with both classical and quantum-enhanced components. The codebase experiments with variance-reduced Monte Carlo sampling, LSTM-based volatility forecasting, and a hybrid quantum-classical LSTM variant.

## High-level summary

- `LastQModel.py` — Main portfolio PFE/calculation engine. Implements:
  - Classical Monte Carlo with improved sampling (Sobol sequences + antithetic variates) and Latin Hypercube sampling for correlated asset simulation.
  - Multi-asset portfolio parsing and correlation handling.
  - Quantum Amplitude Estimation (QAE) integration using Qiskit (Maximum Likelihood Amplitude Estimation) with a classical fallback when Qiskit is not available.
  - Helpers to prepare quantum circuits, optimize for IonQ (via an `AzureWrapper`), and run MLAE or fall back to classical percentiles.

- `lstm_volatility_predictor.py` — Classical LSTM pipeline for volatility forecasting. Implements:
  - Synthetic volatility data generation (mean-reverting process + simulated market events).
  - Feature creation for LSTM (sliding-window lookback sequences).
  - A Keras/TensorFlow LSTM model with training, evaluation, and plotting utilities.
  - A `VolatilityDataGenerator` and `LSTMVolatilityPredictor` classes for modular usage.

- `quantum_lstm_volatility_predictor.py` — Research-level hybrid quantum LSTM.
  - Implements quantum variants of LSTM gates using parameterized quantum circuits (Qiskit).
  - A `QuantumLSTMGate`, `QuantumLSTMCell`, and `QuantumLSTMVolatilityPredictor` that use simulated circuits and sampler primitives.
  - Intended as an experimental comparison point vs classical LSTM. Requires Qiskit and qiskit-machine-learning packages.

- `pfe_convergence.png`, `volatility_predictions.png` — Example output images produced by plotting utilities (if present). These are saved by plotting functions in the LSTM and PFE scripts.

## Contract (inputs / outputs / success criteria)

- Inputs:
  - Portfolio: a dict mapping asset names to dictionaries with keys like `spot`, `strike`, `volatility`, `maturity`, `option_type`, `position`, `notional` (see `MultiAssetPFECalculator.parse_portfolio`).
  - Time series / synthetic volatility arrays (1D numpy arrays) for LSTM training.
  - Optional: `use_hardware=True` and an `AzureWrapper` configured to target IonQ/other hardware for quantum runs.

- Outputs:
  - PFE estimates (classical percentile and quantum MLAE-based estimates when Qiskit is available).
  - Trained LSTM models and prediction arrays (numpy), plus saved plots such as `volatility_predictions.png`.

- Success criteria:
  - Scripts run without errors on a machine with required dependencies.
  - If Qiskit/TensorFlow aren't installed, the code falls back gracefully (classical mode or error messages recommending packages).

## Key behaviors and edge cases

- The project intentionally includes fallbacks when optional dependencies are missing:
  - If Qiskit is missing, quantum features are disabled and classical percentile fallbacks are used (see `QuantumPFECalculator.run_qae_mlae`).
  - If TensorFlow is missing, the LSTM predictor will raise ImportError when instantiated; the code is guarded to print helpful install hints.

- Edge cases to watch for:
  - Very small `num_samples` passed to sampling functions — the code enforces minimum sensible sizes and will warn and adjust.
  - Correlation matrix not positive-definite — code falls back to uncorrelated sampling and prints a warning.
  - Quantum circuits depend on number-of-qubits vs data size; circuits are padded/truncated as needed but be mindful of state-space blow-up.

## Dependencies

Recommended packages (some optional, see notes below):

- Required for core classical functionality:
  - Python 3.8+ (project used modern Python features)
  - numpy
  - scipy
  - matplotlib (optional for plotting)

- For classical ML/LSTM:
  - tensorflow (or tensorflow-cpu)
  - scikit-learn

- For quantum experiments (optional):
  - qiskit, qiskit-aer, qiskit-machine-learning (for quantum LSTM and QAE)
  - azure wrapper (project-specific Azure wrapper used to connect to IonQ via Azure Quantum — optional)

Suggested quick install (into a virtualenv):

```powershell
# create & activate venv
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# install minimal required libs
pip install numpy scipy matplotlib

# for LSTM
pip install tensorflow scikit-learn

# for quantum experiments (optional)
pip install qiskit qiskit-aer qiskit-machine-learning
```

Tip: If you plan to use hardware backends (IonQ via Azure), ensure your `AzureWrapper` is configured and credentials are present; the code expects an `AzureWrapper.get_target_qiskit()` helper.

## How to run (examples)

Below are minimal usage sketches. These are not provided as runnable CLI scripts in the folder, but you can import and run the classes from a Python REPL or small driver script.

1) Quick LSTM data generation + train example (pseudo code):

```python
from QStockPrediction.lstm_volatility_predictor import VolatilityDataGenerator, LSTMVolatilityPredictor

# Generate synthetic data
gen = VolatilityDataGenerator(seed=42)
datasets = gen.generate_training_data({'AAPL': 0.2, 'TSLA': 0.4}, n_days=600, lookback=20)

# Build and train model for AAPL
data = datasets['AAPL']
X, y = data['X'], data['y']
predictor = LSTMVolatilityPredictor(lookback=20)
X_train, X_test, y_train, y_test = predictor.prepare_data(X, y, test_size=0.2)
predictor.build_model()
predictor.train(X_train, y_train, X_test, y_test, epochs=20)
preds = predictor.predict(X_test)
```

2) Running PFE calculator (classical fallback if Qiskit missing):

```python
from QStockPrediction.LastQModel import MultiAssetPFECalculator

portfolio = {
    'AAPL': {'spot': 150, 'strike': 150, 'volatility': 0.2, 'maturity': 0.5, 'option_type': 'call', 'position': 1, 'notional': 1e6},
    'TSLA': {'spot': 200, 'strike': 200, 'volatility': 0.4, 'maturity': 0.5, 'option_type': 'call', 'position': 1, 'notional': 1e6}
}

calc = MultiAssetPFECalculator(confidence_level=0.95, num_qubits_per_asset=2, use_quantum=False)
positions = calc.parse_portfolio(portfolio)
corr = calc.get_correlation_matrix(positions)
samples = calc.sobol_antithetic_sampling(positions, corr, num_samples=1024)
# exposures processing... then call calc.quantum_calculator.run_qae_mlae(exposures)
```

## Where the code already helps you

- Graceful fallbacks exist when optional libs are missing.
- There are utilities for plotting model results and PFE convergence (look for `plot_results` and references to `*_predictions.png`).

## Suggestions / next steps

1. Add a `requirements.txt` or `pyproject.toml` capturing exact versions used for reproducibility.
2. Provide small example scripts (e.g., `examples/run_lstm.py`, `examples/run_pfe.py`) to make the quickstart steps copy-pasteable.
3. Add unit tests for sampling functions and data generation (happy-path + correlation matrix edge-case).
4. Add a short Jupyter notebook that demonstrates: generate data → train classical LSTM → compare to quantum LSTM (simulator).
5. If you plan to run on real quantum hardware, document how to configure `AzureWrapper` and required credentials.

## Notes & caveats

- Quantum LSTM and QAE code are research/experimental: expect long runtimes and additional dependency/version complexity.
- The LSTM implementation uses TensorFlow; if you prefer PyTorch, a port could be made for interoperability or performance comparisons.
