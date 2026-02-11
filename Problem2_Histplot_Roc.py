import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import warnings

mpl.use('TkAgg')  # 可显示图，不可保存图
# mpl.use('Agg')      # 不可显示图，可保存图 （配合plt.savefig使用）

# ===================== 第一步：关闭所有无关警告 =====================
warnings.filterwarnings('ignore')  # 屏蔽matplotlib/tkinter的字体/字形警告
# ===================== 中宋英Times New Roman  =====================
mpl.rcParams.update(mpl.rcParamsDefault)  # 清除缓存，重置默认配置
# 1. 基础配置：中文宋体，解决负号，关闭LaTeX
plt.rcParams['font.sans-serif'] = ['SimSun']  # 中文强制宋体（Windows自带，无兼容问题）
plt.rcParams['axes.unicode_minus'] = False    # 负号正常显示
plt.rcParams['text.usetex'] = False           # 关闭LaTeX，避免干扰字体
# 2. 关键：设置衬线字体为Times New Roman，同时让matplotlib优先调用衬线字体

# 全局字体族设为衬线（关联Times New Roman）
plt.rcParams['font.family'] = 'serif' #看图的中文时注释，显示英文字体时取消注释        

plt.rcParams['font.serif'] = ['Times New Roman']  # 英文Times New Roman，兜底宋体
# 3. 可选：统一字体大小，让图表更整洁
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
# ==================================================================================

# ================= 配置参数 =================
# 基础常量 (来自题目背景)
M_TOTAL = 100000000          # 总需求: 1亿吨
N_PORTS = 3                  # 太空电梯端口数: 3个
CAPACITY_ELEVATOR = 179000   # 单个电梯年运力 (吨/年)
CAPACITY_ROCKET = 125        # 单枚火箭载荷 (吨/次) - 假设Falcon Heavy
FREQ_ROCKET_MAX = 3472       # 全球火箭最大年发射频次 (次/年)

# 决策变量 (这里假设使用第一问求出的某种【混合方案】作为基准)
# 你需要根据你第一问的实际解来修改这个值
# 假设：我们决定用火箭运 20% 的货，电梯运 80% 的货

X_LAUNCHES = 800000                                          # 需要发射的总次数
M_ELEVATOR = M_TOTAL - X_LAUNCHES * CAPACITY_ROCKET          # 电梯需要运的量

# 蒙特卡洛参数
N_SIM = 10000      # 模拟次数
SEED = 2024        # 随机种子复现用

def run_monte_carlo():
    np.random.seed(SEED)
    print(f"--- 开始蒙特卡洛模拟 (N={N_SIM}) ---")
    print(f"方案设定: 火箭发射 {X_LAUNCHES} 次, 电梯运输 {M_ELEVATOR/1e6:.2f} 百万吨")

    # ================= Step 1 & 2: 生成随机样本 =================
    # 1. 成本随机变量 (单位: 美元)
    # 火箭单次成本 ~ N(6000万, 250万)
    sim_cost_rocket_unit = np.random.normal(60000000, 2500000, N_SIM)
    
    # 电梯年运营成本 ~ N(17.9亿, 500万) -> 注意题目单位换算
    # 179000万美元 = 17.9亿 USD
    sim_cost_elevator_year = np.random.normal(1790000000, 5000000, N_SIM)
    
    # 2. 故障状态变量 (伯努利分布 0/1)
    # 火箭故障 (3%): 1=故障, 0=正常
    sim_fail_rocket = np.random.binomial(1, 0.03, N_SIM)
    
    # 电梯故障 (1%): 1=故障, 0=正常
    sim_fail_elevator = np.random.binomial(1, 0.01, N_SIM)

    # ================= Step 3: 仿真计算循环 =================
    # 初始化结果数组
    total_costs = []
    total_times = []

    for i in range(N_SIM):
        # A. 计算时间 (考虑故障带来的效率折损)
        
        # --- 火箭系统时间 ---
        # 故障逻辑: 效率降低50% -> 意味着年发射频次减半 (FREQ * 0.5)
        # 故障逻辑: 修复时间14天 -> 额外增加 14/365 年
        r_eff = 0.5 if sim_fail_rocket[i] == 1 else 1.0
        r_repair = (14 / 365) if sim_fail_rocket[i] == 1 else 0
        
        # 火箭时间 = (总次数 / 有效年频次) + 修复时间
        time_rocket = (X_LAUNCHES / (FREQ_ROCKET_MAX * r_eff)) + r_repair
        
        # --- 电梯系统时间 ---
        # 故障逻辑: 效率降低20% -> 意味着运力变为 80%
        # 故障逻辑: 修复时间0.1年
        e_eff = 0.8 if sim_fail_elevator[i] == 1 else 1.0
        e_repair = 0.1 if sim_fail_elevator[i] == 1 else 0
        
        # 电梯年总运力 = 3个港口 * 单港运力 * 效率
        total_elevator_capacity = N_PORTS * CAPACITY_ELEVATOR * e_eff
        
        # 电梯时间 = (总货量 / 年有效总运力) + 修复时间
        time_elevator = (M_ELEVATOR / total_elevator_capacity) + e_repair
        
        c_ele_ro = math.ceil(M_ELEVATOR / 125) * sim_cost_rocket_unit[i]
        
        # --- 项目总时间 ---
        # 取决于最慢的那个 (并行短板)
        final_time = max(time_rocket, time_elevator)
        total_times.append(final_time)
        
        # B. 计算成本
        # 火箭总成本 = 次数 * 单次随机成本
        c_rocket_total = X_LAUNCHES * sim_cost_rocket_unit[i]
        
        # 电梯总成本 = 3个港口 * 运营年数(即总时间) * 年运营随机成本
        # 注意: 这里假设只要项目没结束，电梯就需要一直维护运营
        c_elevator_total = N_PORTS * final_time * sim_cost_elevator_year[i] + c_ele_ro
        
        total_costs.append(c_rocket_total + c_elevator_total)

    # 转换为数组方便统计
    total_costs = np.array(total_costs)
    total_times = np.array(total_times)

    # ================= Step 4: 统计分析 =================
    # 1. 成本统计 (单位转为 Trillion/万亿 或 Billion/十亿)
    mean_cost = np.mean(total_costs)
    var_cost_95 = np.percentile(total_costs, 95) # 95%置信度下的上限
    
    # 2. 时间统计
    mean_time = np.mean(total_times)
    var_time_95 = np.percentile(total_times, 95)
    
    print("\n=== 仿真结果统计 ===")
    print(f"平均成本 (Mean Cost): ${mean_cost/1e9:.2f} Billion")
    print(f"成本风险值 (VaR 95%): ${var_cost_95/1e9:.2f} Billion")
    print(f"  -> 风险溢价: {(var_cost_95 - mean_cost)/mean_cost*100:.2f}%")
    
    print(f"\n平均工期 (Mean Time): {mean_time:.2f} Years")
    print(f"工期风险值 (VaR 95%): {var_time_95:.2f} Years")
    
    # ================= 可视化 (直方图) =================
    plt.figure(figsize=(6, 5))
    
    # 成本分布图
    # plt.subplot(1, 2, 1)
    sns.histplot(total_costs/1e12, kde=True, element='bars', edgecolor="#8d466f", linewidth=0.7, color="#f0cfe3", bins=50, alpha = 1)
    plt.axvline(mean_cost/1e12, color="#87c0ca", linestyle='-.', label=f'Mean: {mean_cost/1e12:.1f} T')
    plt.axvline(var_cost_95/1e12, color="#ffb85bff", linestyle='--', label=f'VaR 95%: {var_cost_95/1e12:.1f} T')
    plt.xlim(41,58)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, p:f'{x:.1F}T'))
    plt.title('Total Cost Distribution(Roc)')
    plt.xlabel('Cost (USD)')
    plt.legend()
    plt.savefig('Monte_Carlo_Simulation(Roc).png', dpi=900, bbox_inches='tight')  # 高清保存预测图
    
    # # 时间分布图
    # plt.subplot(1, 2, 2)
    # sns.histplot(total_times, kde=True, color="#eebdcd", bins=50)
    # plt.axvline(mean_time, color="#75569e", linestyle=':', label=f'Mean: {mean_time:.1f}y')
    # plt.axvline(var_time_95, color="#e8bf29", linestyle='-', label=f'VaR 95%: {var_time_95:.1f}y')
    # plt.title('Total Time Distribution (Monte Carlo)')
    # plt.xlabel('Time (Years)')
    # plt.legend()
    
    plt.tight_layout()
    plt.show()

# 运行
if __name__ == "__main__":
    run_monte_carlo()