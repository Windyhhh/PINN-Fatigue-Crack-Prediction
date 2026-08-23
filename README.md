# 🔬 PINN Fatigue Crack Growth Prediction | 基于物理信息神经网络的疲劳裂纹扩展预测系统

> **Physics meets deep learning. A Residual-Attention PINN that predicts fatigue crack growth with only 6.42% error — no experimental data needed.**
>
> 物理与深度学习的碰撞。残差注意力物理信息神经网络（PINN）预测疲劳裂纹扩展，平均误差仅 6.42%——无需实验数据。

---

## 🌟 Why This Project? | 项目亮点

Fatigue crack growth is a critical problem in structural integrity assessment. Traditional methods rely on the **Paris equation** (`da/dN = C(ΔK)^m`), which requires laborious experimental calibration. This project implements a **Physics-Informed Neural Network (PINN)** with **residual connections and multi-head self-attention** that directly encodes the Paris equation PDE as a soft constraint — achieving **6.42% average error** with zero experimental data.

疲劳裂纹扩展是结构完整性评估中的关键问题。传统方法依赖 **Paris 方程**（`da/dN = C(ΔK)^m`），需要繁琐的实验标定。本项目实现了一个带有**残差连接和多头自注意力**的**物理信息神经网络（PINN）**，将 Paris 方程 PDE 直接编码为软约束——在**零实验数据**的情况下达到 **6.42% 的平均误差**。

| Metric | Value |
|--------|-------|
| **Average Prediction Error** | **6.42%** (plotting) / 2.36% (evaluation) |
| **Physical Constraint** | Paris Equation PDE Loss + Data Loss + Boundary Conditions |
| **Architecture** | 3× Residual Blocks (256D) + 4-Head Self-Attention + LayerNorm |
| **Convergence** | Basic at 5K epochs, optimal at 30K epochs |
| **Input** | Crack length `a` (1 μm – 110 μm) |
| **Output** | log₁₀(N) cycles (0 – 2×10⁸) |
| **Training Data** | 92 analytical points + 2500 pseudo points |
| **Experimental Data Required** | **None** 🎉 |

---

## 🏗️ Architecture | 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: crack length a (1D)                 │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              FC Layer (1→256) + tanh + LayerNorm              │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Residual Block 1 (256D)                                │  │
│  │  FC → tanh → LayerNorm → 4-Head Self-Attention → +res  │  │
│  └─────────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Residual Block 2 (256D)                                │  │
│  │  FC → tanh → LayerNorm → 4-Head Self-Attention → +res  │  │
│  └─────────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Residual Block 3 (256D)                                │  │
│  │  FC → tanh → LayerNorm → 4-Head Self-Attention → +res  │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Output: log₁₀(N) cycles (1D)                     │
└─────────────────────────────────────────────────────────────┘
```

### Loss Function | 损失函数

```
Total Loss = PDE_Loss + 10 × Data_Loss + 18 × Analytical_Loss

PDE_Loss:       Enforces da/dN = C(ΔK)^m via automatic differentiation
Data_Loss:      Fits 92 log-uniform analytical data points
Analytical_Loss: Fits 2500 pseudo points for physics consistency
```

---

## 📊 Physical Model | 物理模型

### Paris Equation | Paris 方程

```
da/dN = C · (ΔK)^m

ΔK = Y · Δσ · √(π·a)    (Stress intensity factor range)
```

### Analytical Solution | 解析解

```
N(a) = [2 / ((2-m) · C · (Y·Δσ·√π)^m)] · [a^((2-m)/2) - a₀^((2-m)/2)]
```

### Default Parameters | 默认参数

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Paris constant | C | 1×10⁻¹² | Material fatigue resistance |
| Paris exponent | m | 1.1 | ΔK sensitivity |
| Geometric factor | Y | 1.12 | Crack shape correction |
| Stress amplitude | Δσ | 50 MPa | Cyclic loading range |
| Initial crack | a₀ | 1 μm | Detectable crack size |
| Critical crack | a_c | 110 μm | Failure threshold |

### Key Results | 关键结果

- **At N = 1×10⁸ cycles**: crack length ≈ 29 μm
- **At N = 2×10⁸ cycles**: crack length ≈ 104 μm
- **Total fatigue life** (a₀→a_c): ≈ 2.24×10⁸ cycles

---

## 🚀 Quick Start | 快速开始

### 1. Install | 安装

```bash
pip install -r requirements.txt
# or: pip install torch numpy matplotlib
```

### 2. Train | 训练

```bash
python src/training/run_pinn_residual_attention.py
```

Output:
- Real-time loss tracking (PDE / Data / Analytical)
- Best model saved to `experiments/trained_models/best_pinn_model.pth`

### 3. Visualize | 可视化

```bash
python src/visualization/plot_best_results.py
```

Output:
- `experiments/generated_plots/Best_PINN_N_a_Curve.png` — N-a curves (linear, semi-log, error distribution, statistics)

---

## 📈 Training Convergence | 训练收敛

| Epoch | Loss | Stage |
|-------|------|-------|
| 0 | ~1281 | Initial |
| 5,000 | ~9.77 | **Basic convergence** |
| 10,000 | ~5.56 | Rapid optimization |
| 15,000 | ~3.24 | Near optimal |
| 20,000 | ~2.16 | Fine-tuning |
| 25,000 | ~1.24 | **Optimal state** |
| 30,000 | ~0.8 | Final |

### Learning Rate Schedule | 学习率调度

- **Warmup** (0–2000 epochs): 0 → 0.001 (linear)
- **Cosine annealing** (2000–30000 epochs): 0.001 → 1×10⁻⁶

---

## 📁 Project Structure | 项目结构

```
PINN-Fatigue-Crack-Prediction/
├── src/
│   ├── training/
│   │   └── run_pinn_residual_attention.py   # Main training (8.7KB)
│   └── visualization/
│       └── plot_best_results.py              # Result plotting (9.5KB)
├── experiments/
│   ├── trained_models/
│   │   └── best_pinn_model.pth               # Pre-trained model (3.9MB)
│   └── generated_plots/
│       ├── Best_PINN_N_a_Curve.png           # N-a curve visualization
│       └── Best_PINN_Results.png              # Results summary
├── docs/
│   ├── 使用说明.md                             # Usage guide (Chinese)
│   ├── papers/参考.txt                         # Paper references
│   └── tutorials/博客要求                      # Blog requirements
├── 精品博客.md                                  # Technical blog (65KB)
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🔧 Customization | 自定义参数

To use different materials or loading conditions, modify these parameters in **both** scripts:

```python
# src/training/run_pinn_residual_attention.py AND src/visualization/plot_best_results.py
C, m, Y, Delta_sigma = 1e-12, 1.1, 1.12, 50.0
a0, ac = 1e-6, 1.1e-4
```

**Process**:
1. Modify parameters in both files
2. Retrain: `python src/training/run_pinn_residual_attention.py`
3. Re-plot: `python src/visualization/plot_best_results.py`

---

## 🧠 Using the Trained Model | 使用预训练模型

```python
import torch
import sys
sys.path.insert(0, 'src/training')
from run_pinn_residual_attention import ResidualAttentionPINN

# Load model
model = ResidualAttentionPINN()
model.load_state_dict(torch.load('experiments/trained_models/best_pinn_model.pth', map_location='cpu'))
model.eval()

# Predict: given crack length a → cycles N
a = torch.tensor([[1e-5]], dtype=torch.float32)  # 10 μm
with torch.no_grad():
    log10_N = model(a)
    N = 10 ** log10_N.item()

print(f"Crack {a.item()*1e6:.1f} μm → N = {N:.2e} cycles")
```

---

## 📚 References | 参考文献

1. **Paris Equation**: Paris, P., & Erdogan, F. (1963). *A critical analysis of crack propagation laws.* Journal of Basic Engineering.
2. **PINN**: Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks.* Journal of Computational Physics.
3. **Attention**: Vaswani, A., et al. (2017). *Attention is all you need.* NeurIPS.
4. **ResNet**: He, K., et al. (2016). *Deep residual learning for image recognition.* CVPR.

---

## 📄 License | 许可证

MIT License — free to use, modify, and distribute.

---

<div align="center">

**Built with 🔬 for physics-informed deep learning**

[Report Bug](https://github.com/Windyhhh/PINN-Fatigue-Crack-Prediction/issues) · [Request Feature](https://github.com/Windyhhh/PINN-Fatigue-Crack-Prediction/issues)

</div>
