import pandas as pd
import numpy as np
import math

M          = 100000000
CAPACITY_1 = 179000
N_1        = 3
CAPACITY_2 = 125
I          = 3472
PRICE_1_1  = 1790000000
PRICE_1_2  = 0.9*60000000
PRICE_2    = 60000000

df = pd.read_excel("Data_Raw.xlsx")

# 重命名列
df.rename(columns={"n(火箭发射总次数)": "n"}, inplace=True)

# 新增列
# df["x1"] = df["a"] * M                                        # 太空天梯运输材料吨数
# df["x2"] = (1 - df["a"]) * M                                  # 火箭运输材料吨数
df["time1"] = (M - df["n"] * CAPACITY_2) / (CAPACITY_1 * N_1)   # 太空天梯运输总时间（年）
df["time2"] = df["n"] / I                                       # 火箭运输总时间（年）
df["time"] = df[["time1", "time2"]].max(axis=1)                 # 总运输时间（年）
df["cost1"] = N_1 * df["time1"] * PRICE_1_1 + PRICE_1_2 * ((M - df["n"] * CAPACITY_2) / CAPACITY_2).apply(math.ceil)
df["cost2"] = df["n"] * PRICE_2
df["cost"] = df["cost1"] + df["cost2"]

cost_min, cost_max = np.min(df["cost"]), np.max(df["cost"])
time_min, time_max = np.min(df["time"]), np.max(df["time"])
    
df["norm_cost"] = (df["cost"] - cost_min) / (cost_max - cost_min) if cost_max != cost_min else 0
df["norm_time"] = (df["time"] - time_min) / (time_max - time_min) if time_max != time_min else 0

weights = np.array([0.5, 0.5])    
# 计算综合得分（权重×标准化值，得分越低越优）
df["score"] = weights[0] * df["norm_cost"] + weights[1] * df["norm_time"]

# 保存为CSV文件
df.to_csv("Data_Calculate.csv", index=False)