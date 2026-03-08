AAR-WaveMamba: Spatial-Spectral Attention via Wavelet-Integrated Mamba
This is the official PyTorch implementation of the paper "Spatial-Spectral Attention via Wavelet-Integrated Mamba for Behavior Analysis".

AAR-WaveMamba introduces a novel architecture that synergizes Discrete Wavelet Transform (DWT) and Wavelet Packet Decomposition (WPD) with State Space Models (Mamba). It is specifically designed for high-fidelity behavior recognition from sensor data, capturing transient high-frequency features while maintaining the linear complexity and long-range modeling strengths of Mamba.

✨ Key Features
Wavelet Integration: Incorporates DWT/WPD modules (see models/dwt_classifier.py) to extract multiscale spatial-spectral features.

Mamba Backbone: Utilizes a Mamba-based architecture for efficient sequence modeling with a significantly lower memory footprint than Transformers.

Efficiency Evaluator: Includes a built-in evaluation suite to measure Params (M) and FLOPs (G).

Robust Performance: High precision across complex behavioral classes including Stationary, Running, Eating, Trotting, and Walking.

🛠️ Environment & Installation
The environment is built on Python 3.12 and PyTorch 2.7.0+cu128. Since Mamba requires local kernel compilation, please follow the installation order below.

1. Prerequisites
Ensure you have a GPU with CUDA 12.8 support and the nvcc compiler installed.

2. Standard Dependencies
Bash
# Install core scientific stack
pip install -r requirements.txt
3. Mamba Kernels (Local Compilation)
To ensure the kernels match your local hardware and CUDA version, install mamba-ssm and causal-conv1d from source or as follows:

Bash
pip install causal-conv1d>=1.4.0
pip install mamba-ssm>=2.2.0
📂 Repository Structure
models/: Contains core architecture definitions including MambaWithDWT11.py and dwt_classifier.py.

5zhz/: Specific experiment configurations and the five-fold cross-validation script (training_script.sh) for 5Hz datasets.

train_wave.py: The primary entry point for training and validation.

utils.py: Utility functions for performance metrics and efficiency evaluation.