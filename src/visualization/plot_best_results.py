"""
生成最佳版本的a-N图
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 参数 ====================
C, m, Y, Delta_sigma = 1e-12, 1.1, 1.12, 50.0
a0, ac = 1e-6, 1.1e-4  # a0=1μm, ac≈110μm (对应N=2e8)

# ==================== 注意力机制 ====================
class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(AttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.fc_out = nn.Linear(dim, dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.dim)
        out = self.fc_out(out)
        
        return out

# ==================== 残差注意力网络 ====================
class ResidualAttentionPINN(nn.Module):
    def __init__(self):
        super(ResidualAttentionPINN, self).__init__()
        
        self.fc1 = nn.Linear(1, 256)
        self.ln1 = nn.LayerNorm(256)
        
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.attn1 = AttentionBlock(256, num_heads=4)
        
        self.fc3 = nn.Linear(256, 256)
        self.ln3 = nn.LayerNorm(256)
        self.attn2 = AttentionBlock(256, num_heads=4)
        
        self.fc4 = nn.Linear(256, 256)
        self.ln4 = nn.LayerNorm(256)
        self.attn3 = AttentionBlock(256, num_heads=4)
        
        self.fc_out = nn.Linear(256, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, a):
        x = torch.tanh(self.fc1(a))
        x = self.ln1(x)
        
        residual = x
        x = torch.tanh(self.fc2(x))
        x = self.ln2(x)
        x = self.attn1(x.unsqueeze(0)).squeeze(0)
        x = x + residual
        
        residual = x
        x = torch.tanh(self.fc3(x))
        x = self.ln3(x)
        x = self.attn2(x.unsqueeze(0)).squeeze(0)
        x = x + residual
        
        residual = x
        x = torch.tanh(self.fc4(x))
        x = self.ln4(x)
        x = self.attn3(x.unsqueeze(0)).squeeze(0)
        x = x + residual
        
        x = self.fc_out(x)
        return x

# ==================== 解析解 ====================
def analytical_solution(a, a0, C, m, Y, Delta_sigma, return_log=False):
    """
    Paris方程解析解：da/dN = C(ΔK)^m
    其中 ΔK = Y·Δσ·√(πa)

    积分得到：N = ∫(1/(C·(Y·Δσ·√(πa))^m)) da

    结果：N = (2/(2-m)) · (1/(C·(Y·Δσ·√π)^m)) · (a^((2-m)/2) - a0^((2-m)/2))

    注意：a的单位是m，所以a_np已经是米为单位
    """
    if isinstance(a, torch.Tensor):
        a_np = a.detach().cpu().numpy()
    else:
        a_np = a

    # ΔK的系数部分（不含a）
    K_coeff = Y * Delta_sigma * np.sqrt(np.pi)

    # 积分系数
    coeff = 2.0 / ((2.0 - m) * C * (K_coeff ** m))

    # 计算N（循环次数）
    N = coeff * (a_np**((2-m)/2) - a0**((2-m)/2))
    
    if return_log:
        N = np.log10(N + 1e-10)
    
    return torch.tensor(N, dtype=torch.float32).to(device)

# ==================== 加载模型 ====================
print("加载已训练的最佳模型 'best_pinn_model.pth'...")
model = ResidualAttentionPINN().to(device)
try:
    model.load_state_dict(torch.load('best_pinn_model.pth', map_location=device))
except FileNotFoundError:
    print("\n⚠️ 错误: 未找到 'best_pinn_model.pth' 文件。")
    print("请先运行 'python run_pinn_residual_attention.py' 训练并生成模型文件。")
    exit()

model.eval()
print("模型加载成功！")
print("="*70)

# ==================== 生成N-a图 ====================
print("\n生成N-a图（N作为横坐标，范围0-2e8）...")

# 生成a的测试点（从a0到ac）
a_test_np = np.linspace(a0, ac, 500)

# 计算解析解的N值
K_coeff = Y * Delta_sigma * np.sqrt(np.pi)
coeff = 2.0 / ((2.0 - m) * C * (K_coeff ** m))
exponent = (2 - m) / 2  # 0.45

N_true_np = coeff * (a_test_np**exponent - a0**exponent)

# 使用PINN预测：给定a，预测N
a_for_pred = torch.tensor(a_test_np, dtype=torch.float32).view(-1, 1).to(device)
with torch.no_grad():
    log10_N_pred = model(a_for_pred).cpu().numpy()
    log10_N_pred = np.clip(log10_N_pred, -20, 20)
    N_pred_np = 10 ** log10_N_pred.flatten()

# 计算误差
error = np.abs((N_pred_np - N_true_np) / (N_true_np + 1e-10)) * 100
error = np.clip(error, 0, 1000)
mean_error = np.mean(error)

print(f"平均误差: {mean_error:.2f}%")

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# ==================== 子图1: N-a曲线 (线性坐标) ====================
ax1 = axes[0, 0]
# 只绘制N在0-2e8范围内的点
mask_range = N_true_np <= 2e8
ax1.plot(N_true_np[mask_range], a_test_np[mask_range] * 1e6, 'b-', linewidth=3, label='解析解')
ax1.plot(N_pred_np[mask_range], a_test_np[mask_range] * 1e6, 'r--', linewidth=2, label='PINN预测', alpha=0.8)
ax1.set_xlabel('循环次数 N', fontsize=13, fontweight='bold')
ax1.set_ylabel('裂纹长度 a (μm)', fontsize=13, fontweight='bold')
ax1.set_title('N-a曲线 (线性坐标)', fontsize=15, fontweight='bold')
ax1.set_xlim([0, 2e8])
ax1.legend(fontsize=12, loc='upper left')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))

# ==================== 子图2: N-a曲线 (半对数坐标 - N对数) ====================
ax2 = axes[0, 1]
mask = (N_true_np > 1e5) & (N_true_np <= 2e8)
ax2.semilogx(N_true_np[mask], a_test_np[mask] * 1e6, 'b-', linewidth=3, label='解析解')
ax2.semilogx(N_pred_np[mask], a_test_np[mask] * 1e6, 'r--', linewidth=2, label='PINN预测', alpha=0.8)
ax2.set_xlabel('循环次数 N (对数)', fontsize=13, fontweight='bold')
ax2.set_ylabel('裂纹长度 a (μm)', fontsize=13, fontweight='bold')
ax2.set_title('N-a曲线 (半对数坐标)', fontsize=15, fontweight='bold')
ax2.set_xlim([1e5, 2e8])
ax2.legend(fontsize=12, loc='upper left')
ax2.grid(True, alpha=0.3, which='both', linestyle='--')

# ==================== 子图3: 误差分布 ====================
ax3 = axes[1, 0]
ax3.plot(N_true_np[mask_range], error[mask_range], 'g-', linewidth=2.5)
ax3.axhline(y=mean_error, color='r', linestyle='--', linewidth=3,
            label=f'平均误差 = {mean_error:.2f}%')
ax3.axhline(y=5, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='5%误差线')
ax3.axhline(y=10, color='purple', linestyle=':', linewidth=2, alpha=0.7, label='10%误差线')
ax3.set_xlabel('循环次数 N', fontsize=13, fontweight='bold')
ax3.set_ylabel('相对误差 (%)', fontsize=13, fontweight='bold')
ax3.set_title('误差分布', fontsize=15, fontweight='bold')
ax3.set_xlim([0, 2e8])
ax3.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
ax3.legend(fontsize=12, loc='upper right')
ax3.grid(True, alpha=0.3, linestyle='--')

# ==================== 子图4: 统计信息 ====================
ax4 = axes[1, 1]
ax4.axis('off')

# 计算统计信息
error_5 = np.sum(error < 5) / len(error) * 100
error_10 = np.sum(error < 10) / len(error) * 100
error_20 = np.sum(error < 20) / len(error) * 100

# 显示统计信息
info_text = f"""
{'='*60}
N-a曲线 - 最佳PINN性能统计
{'='*60}

网络架构:
  • 残差注意力网络
  • 3个残差块 (256维)
  • 4头自注意力机制
  • LayerNorm稳定化

训练配置:
  • 训练数据: 92个点
  • 伪数据点: 2500个
  • DataLoss权重: λ = 18
  • 训练轮数: 30000
  • 学习率调度: Warmup + 余弦退火

性能指标:
  • 平均误差: {mean_error:.2f}%
  • 误差<5%的点: {error_5:.1f}%
  • 误差<10%的点: {error_10:.1f}%
  • 误差<20%的点: {error_20:.1f}%

物理参数:
  • Paris常数 C = {C}
  • Paris指数 m = {m}
  • 几何因子 Y = {Y}
  • 应力幅值 Δσ = {Delta_sigma} MPa
  • 初始裂纹 a₀ = {a0*1e6:.2f} μm
  • 临界裂纹 aᶜ = {ac*1e6:.2f} μm

N取值范围: 0 - 2×10⁸ 次循环

说明: 横坐标为循环次数N，纵坐标为裂纹长度a (μm)
{'='*60}
"""

ax4.text(0.05, 0.5, info_text, verticalalignment='center',
         fontsize=10.5, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4, pad=1))

plt.tight_layout()
plt.savefig('Best_PINN_N_a_Curve.png', dpi=300, bbox_inches='tight')
print("\n图表已保存: Best_PINN_N_a_Curve.png")

plt.show()

print("\n" + "="*70)
print("完成！")
print("="*70)

