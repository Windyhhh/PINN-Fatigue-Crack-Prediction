<div align="center">

# 疲劳裂纹扩展预测 | PINN-Fatigue-Crack-Prediction

### PINN-based fatigue crack growth prediction.

Physics-informed neural network with Paris-law constraints — ~6.42% average prediction error from sparse data.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>

---

**PINN-Fatigue-Crack-Prediction** predicts **fatigue crack growth** with a **physics-informed neural network** constrained by the **Paris law**, achieving **~6.42% average error** vs analytical solutions — from just 92 training points.

> [!NOTE]
> 中文项目：基于物理信息神经网络（PINN）的疲劳裂纹扩展预测——Paris 方程物理约束，平均误差 6.42%，仅 92 个数据点。

---

## Features

- **PINN model** — deep learning with physics constraints.
- **Paris-law constraint** — physics-informed crack-growth modeling.
- **Data-efficient** — 92 sparse training points (log-uniform).
- **Accurate** — ~6.42% average error vs analytical solution.
- **Modular** — easy to tune physical params & network structure.

---

## Quickstart

```bash
git clone https://github.com/Windyhhh/PINN-Fatigue-Crack-Prediction.git
cd PINN-Fatigue-Crack-Prediction

pip install -r requirements.txt

python src/train.py          # train the PINN
python src/predict.py        # predict crack growth
```

---

## Project Structure

```
PINN-Fatigue-Crack-Prediction/
├── src/                    # PINN model, training, prediction
├── data/                   # sparse training points
├── configs/                # physical params
└── docs/                   # usage, blog
```

---


## Results

<div align="center">
  <img src="experiments/generated_plots/Best_PINN_N_a_Curve.png" alt="Crack growth N-a curve (analytical vs PINN)" width="70%"/>
  <img src="experiments/generated_plots/Best_PINN_Results.png" alt="PINN prediction results" width="70%"/>
</div>

---
## License

MIT — free to use, modify and distribute.
