import pandas as pd
import numpy as np
import math

# ================= 1. 参数与限制设置 =================
# 基础数据
total_annual_demand = 613200
monthly_need = total_annual_demand / 12  # 每月需水 51100 吨

# 比例系数
loss_rate = 0.05       # 损耗 5%
lunar_ratio = 0.10     # 月球取水 10%
backup_ratio = 0.10    # 备用水 10%

# 成本参数 (USD)
cost_recycle_per_ton = 500
cost_lunar_per_ton = 2

# 运输成本参数
# 太空电梯 Step 1: 10,000 USD/吨
se_step1_cost_per_ton = 10000 
# 太空电梯 Step 2: 火箭转运 (容量125吨, 5400万/次)
se_step2_capacity = 125
se_step2_cost_per_launch = 54000000 

# 地球火箭 (容量125吨, 6000万/次)
rocket_capacity = 125
rocket_cost_per_launch = int(54000000 / 0.9) # 60,000,000

# ★★★ 新增：每月运力限制 ★★★
SE_MONTHLY_LIMIT = 44750
ROCKET_MONTHLY_LIMIT = 36125

# ================= 2. 辅助计算函数 =================

def get_eta(t):
    """根据月份计算水回收率 η (0% -> 90%)"""
    if 1 <= t <= 2:
        return 0.0 
    else: # 3-12月
        val = 0.20 + (t - 3) * 0.10
        return min(val, 0.90) 

def calc_se_cost(mass):
    """计算太空电梯成本 (质量成本 + 转运发射成本)"""
    if mass <= 0: return 0
    # Step 1: 质量费
    step1 = mass * se_step1_cost_per_ton
    # Step 2: 发射费 (次数向上取整)
    launches = math.ceil(mass / se_step2_capacity)
    step2 = launches * se_step2_cost_per_launch
    return step1 + step2

def calc_rocket_cost(mass):
    """计算地球火箭成本 (发射费)"""
    if mass <= 0: return 0
    launches = math.ceil(mass / rocket_capacity)
    return launches * rocket_cost_per_launch

# ================= 3. 逐月计算 =================
results = []

for t in range(1, 13):
    eta = get_eta(t)
    
    # --- 水量平衡 ---    
    q_recycled = eta * monthly_need
    q_lunar = lunar_ratio * monthly_need
    q_use = monthly_need - q_recycled - q_lunar
    q_backup = backup_ratio * q_use
    
    # 总运输需求 Q_gross (含5%损耗)
    q_transport_net = q_use + q_backup
    q_transport_gross = q_transport_net / (1 - loss_rate)
    q_loss = q_transport_gross * loss_rate
    
    # --- 成本计算 ---
    cost_recycle = q_recycled * cost_recycle_per_ton
    cost_lunar = q_lunar * cost_lunar_per_ton
    
    # --- 运输方案 (应用限制) ---
    
    # 方案1：优先用太空电梯
    # 逻辑：电梯运量 = min(总需求, 电梯上限 44750)
    #       火箭运量 = 总需求 - 电梯运量
    mass_se_s1 = min(q_transport_gross, SE_MONTHLY_LIMIT)
    mass_rocket_s1 = max(0, q_transport_gross - mass_se_s1)
    
    cost_trans_s1 = calc_se_cost(mass_se_s1) + calc_rocket_cost(mass_rocket_s1)
    
    # 方案2：优先用地球火箭
    # 逻辑：火箭运量 = min(总需求, 火箭上限 36125)
    #       电梯运量 = 总需求 - 火箭运量
    mass_rocket_s2 = min(q_transport_gross, ROCKET_MONTHLY_LIMIT)
    mass_se_s2 = max(0, q_transport_gross - mass_rocket_s2)
    
    cost_trans_s2 = calc_rocket_cost(mass_rocket_s2) + calc_se_cost(mass_se_s2)
    
    # 方案3：各运一半
    # 逻辑：两边各分担一半。
    # 检查：最大月运量约5.4万吨，减半为2.7万吨，均未超过各自上限(3.6万, 4.4万)，故无需特殊溢出处理。
    mass_se_s3 = q_transport_gross / 2
    mass_rocket_s3 = q_transport_gross / 2
    
    cost_trans_s3 = calc_se_cost(mass_se_s3) + calc_rocket_cost(mass_rocket_s3)
    
    # 计算各方案总成本
    total_cost_s1 = cost_recycle + cost_lunar + cost_trans_s1
    total_cost_s2 = cost_recycle + cost_lunar + cost_trans_s2
    total_cost_s3 = cost_recycle + cost_lunar + cost_trans_s3
    
    results.append({
        "Month": t,
        "Recycle_Rate(eta)": round(eta, 2),
        "Q_recycled": round(q_recycled, 2),
        "Q_lunar": round(q_lunar, 2),
        "Q_use": round(q_use, 2),
        "Q_backup": round(q_backup, 2),
        "Q_loss": round(q_loss, 2),
        "Total_Transport_Demand": round(q_transport_gross, 2),
        # 方案1数据
        "S1_SE_Mass": round(mass_se_s1, 2),
        "S1_Rocket_Mass": round(mass_rocket_s1, 2),
        "S1_Total_Cost": round(total_cost_s1, 2),
        
        # 方案2数据
        "S2_SE_Mass": round(mass_se_s2, 2),
        "S2_Rocket_Mass": round(mass_rocket_s2, 2),
        "S2_Total_Cost": round(total_cost_s2, 2),
        
        # 方案3数据
        "S3_SE_Mass": round(mass_se_s3, 2),
        "S3_Rocket_Mass": round(mass_rocket_s3, 2),
        "S3_Total_Cost": round(total_cost_s3, 2)
    })

# ================= 4. 汇总与保存 =================
df = pd.DataFrame(results)

# 打印年度汇总
print("=== 年度总成本汇总 (美元) ===")
print(f"方案一 (优先电梯): {df['S1_Total_Cost'].sum():,.2f}")
print(f"方案二 (优先火箭): {df['S2_Total_Cost'].sum():,.2f}")
print(f"方案三 (混合运输): {df['S3_Total_Cost'].sum():,.2f}")

# 保存
df.to_csv('Problem3_Detailed_Data.csv', index=False, encoding='utf_8_sig')
print("\n前5个月详细数据预览:")
print(df[['Month', 'Q_recycled', 'Q_lunar', 'Q_use', 'Q_backup', 'Q_loss', 'Total_Transport_Demand', 'S1_SE_Mass', 'S1_Rocket_Mass', 'S2_SE_Mass', 'S2_Rocket_Mass']].head())