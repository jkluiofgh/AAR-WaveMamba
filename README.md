# AAR-WaveMamba

This repository is an official PyTorch implementation of the paper **"Spatial-Spectral Attention via Wavelet-Integrated Mamba for Behavior Analysis"**. 

**AAR-WaveMamba** introduces a novel architecture that synergizes Discrete Wavelet Transform (DWT) and Wavelet Packet Decomposition (WPD) with State Space Models (Mamba). It captures transient high-frequency features from sensor data while maintaining the linear complexity and long-range modeling strengths of Mamba.

---

## Requirements

Since Mamba's core operators (e.g., `selective_scan`) require local CUDA compilation, a simple `pip install -r requirements.txt` might not work out-of-the-box across different machines. 

This is our exact experiment environment for your reference:
* python 3.12
* pytorch 2.7.0 + cuda 12.8
* `mamba-ssm` >= 2.2.0 (compiled locally)
* `causal-conv1d` >= 1.4.0 (compiled locally)
* `PyWavelets` 1.8.0

*Note: If you encounter installation errors, please ensure your `nvcc` compiler version matches your PyTorch CUDA version, and install `mamba-ssm` and `causal-conv1d` from source according to their official guidelines.*

---

## Details

### 1. Dataset

We used a public dataset containing sensor-based animal behavior data, which is available at:
[https://lifesciences.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/dans-xhn-bsfb](https://lifesciences.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/dans-xhn-bsfb)

In our experiment, the raw sensor signals are pre-processed and sampled at 5Hz. The standardized data is divided into training, validation, and testing sets circularly according to a strict **five-fold cross-validation** protocol to evaluate the model's robustness across different subjects.

### 2. Train the model

All of the training, validation, testing, and efficiency evaluation processes are integrated. You can easily execute the 5-fold cross-validation by running the provided shell script:

```bash
cd 5zhz
bash training_script.sh