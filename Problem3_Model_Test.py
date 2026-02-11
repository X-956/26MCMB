import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
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
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 16
# ==================================================================================


# ================= 1. 核心模型类 (Core Logic) =================
class WaterSupplyModel:
    def __init__(self, recycle_cost_per_ton=500, population_cnt=100000, max_eta=0.90):
        # 基础参数
        self.base_demand = 613200  # 基准年需水量
        self.population_cnt = population_cnt
        # 假设需水量与人口成正比
        self.total_annual_demand = self.base_demand * (population_cnt / 100000.0)
        self.monthly_need = self.total_annual_demand / 12
        
        self.max_eta = max_eta      # 新增：最大回收率上限参数

        self.loss_rate = 0.05       # 损耗
        self.lunar_ratio = 0.10     # 月球取水比例
        self.backup_ratio = 0.10    # 备用水比例
        
        self.cost_recycle_per_ton = recycle_cost_per_ton
        self.cost_lunar_per_ton = 2
        
        # 运输参数
        self.se_step1_cost = 10000
        self.se_step2_cap = 125
        self.se_step2_cost = 54000000
        
        self.rocket_cap = 125
        self.rocket_cost = 60000000
        
        # 运力限制
        self.SE_LIMIT = 44750
        self.ROCKET_LIMIT = 36125
        
    def get_eta(self, t):
        """计算回收率，考虑上限 max_eta"""
        if 1 <= t <= 2:
            return 0.0
        else: # 3-12月
            # 正常增长逻辑：3月0.2 ... 10月0.9
            val = 0.20 + (t - 3) * 0.10
            # 取计算值与设定的最大上限的较小值
            return min(val, self.max_eta)

    def calc_se_cost(self, mass):
        """太空电梯成本计算 (含离散发射次数)"""
        if mass <= 0: return 0
        step1 = mass * self.se_step1_cost
        launches = math.ceil(mass / self.se_step2_cap)
        step2 = launches * self.se_step2_cost
        return step1 + step2

    def calc_rocket_cost(self, mass):
        """地球火箭成本计算 (含离散发射次数)"""
        if mass <= 0: return 0
        launches = math.ceil(mass / self.rocket_cap)
        return launches * self.rocket_cost

    def run_monthly_simulation(self):
        """运行12个月的模拟"""
        results = []
        for t in range(1, 13):
            eta = self.get_eta(t)
            
            q_recycled = eta * self.monthly_need
            q_lunar = self.lunar_ratio * self.monthly_need
            q_use = self.monthly_need - q_recycled - q_lunar
            q_backup = self.backup_ratio * q_use
            
            # 运输总需求 (Gross)
            q_trans_net = q_use + q_backup
            q_trans_gross = q_trans_net / (1 - self.loss_rate)
            q_loss = q_trans_gross * self.loss_rate
            
            # 固定成本 (回收+月球)
            c_recycle = q_recycled * self.cost_recycle_per_ton
            c_lunar = q_lunar * self.cost_lunar_per_ton
            
            # --- 方案1: 优先太空电梯 ---
            m_se_1 = min(q_trans_gross, self.SE_LIMIT)
            m_ro_1 = max(0, q_trans_gross - m_se_1)
            c_trans_1 = self.calc_se_cost(m_se_1) + self.calc_rocket_cost(m_ro_1)
            
            # --- 方案2: 优先地球火箭 ---
            m_ro_2 = min(q_trans_gross, self.ROCKET_LIMIT)
            m_se_2 = max(0, q_trans_gross - m_ro_2)
            c_trans_2 = self.calc_rocket_cost(m_ro_2) + self.calc_se_cost(m_se_2)
            
            # --- 方案3: 混合 (50/50) ---
            m_se_3 = q_trans_gross / 2
            m_ro_3 = q_trans_gross / 2
            c_trans_3 = self.calc_se_cost(m_se_3) + self.calc_rocket_cost(m_ro_3)
            
            results.append({
                "Month": t,
                "Eta": eta,
                "Transport_Demand": q_trans_gross,
                "Q_recycled": round(q_recycled, 2),
                "Q_lunar": round(q_lunar, 2),
                "Q_use": round(q_use, 2),
                "Q_backup": round(q_backup, 2),
                "Q_loss": round(q_loss, 2),
                # 方案1数据
                "S1_SE_Mass": m_se_1,
                "S1_Rocket_Mass": m_ro_1,
                "Cost_S1": c_recycle + c_lunar + c_trans_1,
                # 方案2数据
                "S2_SE_Mass": m_se_2,
                "S2_Rocket_Mass": m_ro_2,
                "Cost_S2": c_recycle + c_lunar + c_trans_2,
                
                # 方案3数据
                "S3_SE_Mass": m_se_3,
                "S3_Rocket_Mass": m_ro_3,
                "Cost_S3": c_recycle + c_lunar + c_trans_3
            })
        return pd.DataFrame(results)

# ================= 2. 执行检验与数据保存 =================

# --- 2.1 可行性验证 (Base Case: Max Eta = 0.9) ---
print("正在进行可行性验证...")
model_base = WaterSupplyModel(max_eta=0.90)
df_base = model_base.run_monthly_simulation()
df_base.to_csv('Validation_Feasibility1.csv', index=False)

# --- 2.2 鲁棒性检验: 水回收成本波动 (450-550) ---
print("正在进行成本鲁棒性检验...")
cost_range = np.arange(450, 551, 5) 
res_cost = []
for c in cost_range:
    m = WaterSupplyModel(recycle_cost_per_ton=c, max_eta=0.90)
    df = m.run_monthly_simulation()
    res_cost.append({
        "Recycle_Cost": c,
        "Total_S1": df['Cost_S1'].sum(),
        "Total_S2": df['Cost_S2'].sum(),
        "Total_S3": df['Cost_S3'].sum()
    })
df_robust_cost = pd.DataFrame(res_cost)
df_robust_cost.to_csv('Validation_Robustness_RecycleCost.csv', index=False)

# --- 2.3 鲁棒性检验: 人口波动 (9万-11万) ---
print("正在进行人口鲁棒性检验...")
pop_range = np.arange(90000, 110001, 1000) 
res_pop = []
for p in pop_range:
    m = WaterSupplyModel(population_cnt=p, max_eta=0.90)
    df = m.run_monthly_simulation()
    res_pop.append({
        "Population": p,
        "Total_S1": df['Cost_S1'].sum(),
        "Total_S2": df['Cost_S2'].sum(),
        "Total_S3": df['Cost_S3'].sum()
    })
df_robust_pop = pd.DataFrame(res_pop)
df_robust_pop.to_csv('Validation_Robustness_Population.csv', index=False)

# --- 2.4 (新增) 鲁棒性检验: 最大回收率波动 (81%-90%) ---
print("正在进行回收率上限鲁棒性检验...")
# 生成 0.81 到 0.90 的序列
eta_range = np.arange(0.81, 0.9001, 0.01) 
res_eta = []
for e in eta_range:
    m = WaterSupplyModel(max_eta=e)
    df = m.run_monthly_simulation()
    res_eta.append({
        "Max_Eta": e,
        "Total_S1": df['Cost_S1'].sum(),
        "Total_S2": df['Cost_S2'].sum(),
        "Total_S3": df['Cost_S3'].sum()
    })
df_robust_eta = pd.DataFrame(res_eta)
df_robust_eta.to_csv('Validation_Robustness_MaxEta1.csv', index=False)


# ================= 3. 可视化展示 (Visual Check) =================

# 图1: 可行性
plt.figure(figsize=(12, 6))
plt.bar(df_base['Month'], df_base['Transport_Demand'], color='skyblue', label='Transport Demand')
plt.axhline(y=model_base.SE_LIMIT, color='orange', linestyle='--', linewidth=2, label='SE Limit')
plt.axhline(y=model_base.ROCKET_LIMIT, color='green', linestyle='--', linewidth=2, label='Rocket Limit')
plt.title('Feasibility Check')
plt.xlabel('Month')
plt.ylabel('Mass (Tons)')
plt.legend(loc = 'right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Plot_Feasibility1.png')

# 图2: 鲁棒性 - 回收成本
plt.figure(figsize=(10, 6))
plt.plot(df_robust_cost['Recycle_Cost'], df_robust_cost['Total_S1'], label='Scheme 1')
plt.plot(df_robust_cost['Recycle_Cost'], df_robust_cost['Total_S2'], label='Scheme 2')
plt.plot(df_robust_cost['Recycle_Cost'], df_robust_cost['Total_S3'], label='Scheme 3')
plt.title('Robustness: Recycling Cost')
plt.ylabel('Cost ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Plot_Robustness_Cost1.png')

# 图3: 鲁棒性 - 人口
plt.figure(figsize=(10, 6))
plt.plot(df_robust_pop['Population'], df_robust_pop['Total_S1'], label='Scheme 1')
plt.plot(df_robust_pop['Population'], df_robust_pop['Total_S2'], label='Scheme 2')
plt.plot(df_robust_pop['Population'], df_robust_pop['Total_S3'], label='Scheme 3')
plt.title('Robustness: Population')
plt.ylabel('Cost ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Plot_Robustness_Population1.png')

# 图4: (新增) 鲁棒性 - 最大回收率
plt.figure(figsize=(10, 6))
plt.plot(df_robust_eta['Max_Eta'], df_robust_eta['Total_S1'], marker='o', label='Scheme 1')
plt.plot(df_robust_eta['Max_Eta'], df_robust_eta['Total_S2'], marker='s', label='Scheme 2')
plt.plot(df_robust_eta['Max_Eta'], df_robust_eta['Total_S3'], marker='^', label='Scheme 3')
plt.title('Robustness: Max Recycling Rate (Eta)')
plt.xlabel('Max Recycling Rate')
plt.ylabel('Annual Cost ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Plot_Robustness_Eta1.png')
plt.show()
print("完成。")