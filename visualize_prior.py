import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import sys
from pathlib import Path

# 将 src 目录加入路径
sys.path.append(str(Path(__file__).resolve().parent))

try:
    from src.model import build_model
    from src.utils import load_config, build_v_prior_from_config, get_device
except ImportError:
    print("❌ 错误：无法导入 src 目录下的模块。")
    sys.exit(1)

# ===================== 配置区域 =====================
CONFIG_PATH = "configs/hh_swin.yaml"
CHECKPOINT_PATH = r"checkpoints\model_1.pth" 

OUTPUT_FIG_NAME = "Figure_Learned_Soft_Prior_RedBlue_Square_Unclamped_ThinnerLine.png" 
WAVENUMBER_RANGE = (200, 1600)


COLOR_POS = '#C00000' # 深红色 (对应向上, H-A态)
COLOR_NEG = '#00A0E9' # 亮蓝色 (对应向下, H-HA态)
# ===================================================

def visualize_learned_state_decoupled_prior():
    device = get_device()
 
    
    config = load_config(CONFIG_PATH)
    v_hard_prior_raw = build_v_prior_from_config(config) 
    
    model = build_model(config['model']).to(device)
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    except FileNotFoundError:

        sys.exit(1)

    with torch.no_grad():
        res_pos = model.res_pos.detach().cpu()
        res_neg = model.res_neg.detach().cpu()
        residual_scale = model.prior_residual_scale 
        
        aligned_base_raw = model.align_prior_to_token_length(v_hard_prior_raw, res_pos.shape[0]).cpu()
        
        # 同步模型内部的 0.5 缩放
        scaled_base_raw = aligned_base_raw * 0.5
        
        base_pos_interp = torch.nn.functional.relu(scaled_base_raw)
        base_neg_interp = torch.nn.functional.relu(-scaled_base_raw)
        
        learned_pos = base_pos_interp + residual_scale * torch.tanh(res_pos)
        learned_neg = base_neg_interp + residual_scale * torch.tanh(res_neg)
        
        m_pos_soft = torch.clamp(learned_pos, max=1.0)
        m_neg_soft = torch.clamp(learned_neg, max=1.0)
        # ========================================================
        
        v_soft_prior_signed = (m_pos_soft - m_neg_soft).numpy()
        base_raw_interp_numpy = scaled_base_raw.numpy() 
        
    x_axis = np.linspace(WAVENUMBER_RANGE[0], WAVENUMBER_RANGE[1], v_soft_prior_signed.shape[0])

    print("🎨 正在绘制正方形 Figure...")
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix" 

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.subplots_adjust(hspace=0.2) 

    # ---------- 上子图 ax1: 软锚点先验 ----------
    ax1.fill_between(x_axis, 0, base_raw_interp_numpy, where=(base_raw_interp_numpy > 0), interpolate=True,
                     color=COLOR_POS, alpha=0.35, label=r'H-A Soft Anchor ($P = +0.5$)') 
    ax1.fill_between(x_axis, 0, base_raw_interp_numpy, where=(base_raw_interp_numpy < 0), interpolate=True,
                     color=COLOR_NEG, alpha=0.35, label=r'H-HA Soft Anchor ($P = -0.5$)') 
    
    ax1.step(x_axis, base_raw_interp_numpy, where='mid', color='grey', linewidth=1.8, linestyle='--')

    ax1.set_ylim(-1.15, 1.15)
    ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax1.set_yticklabels(['-1.0', '-0.5', '0.0', '+0.5', '+1.0'])
    ax1.set_ylabel(r'$\mathbf{P_{external}^{anchor}}$', fontsize=18)
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax1.grid(True, linestyle=':', alpha=0.45)
    
    ax1.set_title("Soft Anchoring Allows AI to Amplify Core Features and Discover Hidden Signals", 
                  fontsize=18, fontweight='bold', pad=18)

    # ---------- 下子图 ax2: AI学到的曲线 ----------
    ax2.fill_between(x_axis, 0, v_soft_prior_signed, where=(v_soft_prior_signed > 0), interpolate=True, color=COLOR_POS, alpha=0.8) 
    ax2.fill_between(x_axis, 0, v_soft_prior_signed, where=(v_soft_prior_signed < 0), interpolate=True, color=COLOR_NEG, alpha=0.8) 

    # ===================== 修改点 =====================
    # 将 linewidth 从 2.5 减小到 1.5
    ax2.plot(x_axis, v_soft_prior_signed, color='black', linewidth=1, linestyle='-', label='Learned AI Soft Prior')
    # =================================================

    ax2.axhline(y=1.0, color='grey', linewidth=1.0, linestyle=':')
    ax2.axhline(y=-1.0, color='grey', linewidth=1.0, linestyle=':')
    ax2.axhline(y=0.5, color='grey', linewidth=1, linestyle='--', alpha=0.5)
    ax2.axhline(y=-0.5, color='grey', linewidth=1, linestyle='--', alpha=0.5)

    ax2.fill_between(x_axis, -0.2, 0.2, color='grey', alpha=0.1, hatch='//', label='Awakened Zone (Proven Gradient Flow)')

    ax2.set_ylim(-1.15, 1.15)
    ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax2.set_yticklabels(['-1.0', '-0.5', '0.0', '+0.5', '+1.0'])
    ax2.set_ylabel(r'$\mathbf{P_{chem}^{final}}$', fontsize=18)
    ax2.set_xlabel('Raman Shift ($cm^{-1}$)', fontsize=19, fontweight='bold')
    
    ax2.xaxis.set_major_locator(MultipleLocator(200))
    ax2.xaxis.set_minor_locator(MultipleLocator(50))
    ax2.tick_params(axis='x', which='major', length=8, width=1.8)
    
    ax2.grid(True, linestyle=':', alpha=0.45)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    
    fig.legend(handles1 + handles2, labels1 + labels2, 
               loc='lower center', ncol=3, fontsize=13, frameon=True, shadow=True, 
               bbox_to_anchor=(0.5, 0.02)) 

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20, top=0.92, left=0.12, right=0.95)
    
    plt.savefig(OUTPUT_FIG_NAME, dpi=400) 
    print(f"✅ 线条变细的 Figure 已保存为: {OUTPUT_FIG_NAME}")
    plt.show()

if __name__ == "__main__":
    visualize_learned_state_decoupled_prior()