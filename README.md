<div align="center">

# 🔥 PINN-Fatigue-Crack-Prediction

### Physics-informed neural network for fatigue crack prediction.

Residual-attention PINN — 6.42% error on crack-growth prediction.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>

---

**PINN-Fatigue-Crack-Prediction** predicts fatigue crack growth with a **residual-attention PINN**, achieving **6.42% error**. It couples physics-informed losses with an attention mechanism for better extrapolation.

> [!NOTE]
> 中文项目：物理信息神经网络疲劳裂纹预测——残差注意 PINN，误差 6.42%。

---

## Quickstart

```bash
git clone https://github.com/Windyhhh/PINN-Fatigue-Crack-Prediction.git
cd PINN-Fatigue-Crack-Prediction

pip install -r requirements.txt
pip install -e .

# train the residual-attention PINN
python src/training/run_pinn_residual_attention.py

# visualize the best results
python src/visualization/plot_best_results.py
```

A trained model is included at `experiments/trained_models/best_pinn_model.pth`.

---

## Features

- **Residual-attention PINN** — physics-informed + attention.
- **High accuracy** — 6.42% error.
- **Trained model** — pre-trained weights + plotting.

---

## Project Structure

```
PINN-Fatigue-Crack-Prediction/
├── src/
│   ├── training/run_pinn_residual_attention.py
│   └── visualization/plot_best_results.py
├── experiments/
│   ├── trained_models/best_pinn_model.pth
│   └── generated_plots/
├── docs/papers/
└── setup.py
```

---

## License

MIT — free to use, modify and distribute.
