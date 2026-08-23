# ⚡ PINN 疲劳裂纹预测 | Physics-Informed Neural Network for Fatigue Crack

> **用神经网络求解断裂力学 PDE，物理约束嵌入损失函数——数据稀缺也能精准预测裂纹扩展。**
>
> *Embed fracture mechanics PDEs into neural network loss — accurate crack growth prediction even with scarce data.*

---

## ⭐ 核心卖点 | Why Star This

| 卖点 | Feature | 一句话 |
|------|---------|--------|
| 🔬 **物理嵌入** | Physics-Informed | 断裂力学 PDE 直接作为损失约束，不是纯数据驱动 |
| 📉 **数据高效** | Data-Efficient | 小样本场景下碾压纯数据驱动方法 |
| 🦴 **裂纹扩展** | Crack Growth | 预测疲劳裂纹长度随循环次数的演化 |
| 🎯 **可解释** | Interpretable | 物理约束提供可解释性，不是黑箱 |
| 📊 **完整实验** | Full Experiments | 多种材料参数下的对比实验 |

---

## 🏆 技术栈 | Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red?logo=pytorch)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-orange?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-green?logo=plotly)

---

## 📊 方法对比 | Method Comparison

| 方法 | 数据需求 | 物理一致性 | 可解释性 | 小样本表现 |
|------|---------|-----------|---------|-----------|
| 纯数据驱动 NN | 🔴 高 | ❌ 无 | ❌ 黑箱 | 🐢 差 |
| 有限元 FEM | 🟡 中 | ✅ 强 | ✅ 强 | 🟡 中 |
| **PINN (本项目)** | 🟢 低 | ✅ 强 | ✅ 强 | 🚀 好 |

---

## 🚀 快速开始 | Quick Start

```bash
git clone https://github.com/Windyhhh/PINN-Fatigue-Crack-Prediction.git
cd PINN-Fatigue-Crack-Prediction
pip install -r requirements.txt
python train.py --material steel --cycles 10000
```

---

## 📂 项目结构 | Project Structure

```
PINN-Fatigue-Crack-Prediction/
├── train.py                   # 训练入口
├── requirements.txt           # 依赖
├── models/
│   └── pinn.py                # PINN 模型定义
├── physics/
│   └── fracture_mechanics.py  # 断裂力学 PDE
├── data/                      # 实验数据
├── utils/                     # 工具函数
└── results/                   # 预测结果
```

---

## 🔬 核心原理 | Core Idea

### 物理信息神经网络 | Physics-Informed Neural Network

PINN 将控制方程（PDE）作为软约束嵌入损失函数：

```
L_total = L_data + λ · L_PDE

L_data = MSE(NN(x) - y_observed)      # 数据拟合损失
L_PDE  = MSE(PDE_residual(NN(x)))      # PDE 残差损失
```

### 断裂力学约束 | Fracture Mechanics

- **Paris 定律**：da/dN = C(ΔK)^m — 裂纹扩展速率
- **应力强度因子**：K = σ√(πa) — 裂纹尖端应力场
- **边界条件**：初始裂纹长度、循环载荷

---

## 🎯 应用场景 | Use Cases

- 🏗️ **结构健康监测**：桥梁、飞机结构的疲劳寿命预测
- 🚗 **汽车工业**：发动机部件、底盘的疲劳分析
- ✈️ **航空航天**：机翼、机身结构的裂纹扩展预测
- 🏭 **制造业**：压力容器、管道的安全评估

---

## 📚 参考文献 | References

- Raissi, M., et al. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." JCP 2019.
- Paris, P., & Erdogan, F. "A critical analysis of crack propagation laws." Journal of Basic Engineering 1963.

---

## 📄 License

MIT License — 自由使用、修改和分发。

---

> 💡 **物理 + AI 的交叉创新，Star ⭐ 一下支持开源！**
