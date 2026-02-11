import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

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
# plt.rcParams['font.family'] = 'serif' #看图的中文时注释，显示英文字体时取消注释        

plt.rcParams['font.serif'] = ['Times New Roman']  # 英文Times New Roman，兜底宋体
# 3. 可选：统一字体大小，让图表更整洁
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
# ==================================================================================

# 常量定义
M          = 100000000
CAPACITY_1 = 179000
N_1        = 3
CAPACITY_2 = 125
I          = 3472
PRICE_1_1  = 1786710445.56
PRICE_1_2  = 0.9*64260671.50
PRICE_2    = 64260671.50
CVST       = 1

POP_SIZE   = 100
SEED       = None
GEN_T      = 100


# ===================== 1. AHP权重计算模块 =====================
def ahp_calculate_weight(judgment_matrix):
    """
    层次分析法(AHP)计算权重，包含一致性检验
    :param judgment_matrix: 两两比较判断矩阵（n×n，这里n=2，对应成本、耗时）
    :return: 归一化权重（若一致性检验通过），否则提示调整矩阵
    """
    # 步骤1：计算判断矩阵的特征值和特征向量
    eig_vals, eig_vecs = np.linalg.eig(judgment_matrix)
    # 找到最大特征值对应的特征向量
    max_eig_val = np.max(eig_vals)
    max_eig_vec = eig_vecs[:, np.argmax(eig_vals)].real  # 取实部（避免复数）
    
    # 步骤2：特征向量归一化，得到初始权重
    weights = max_eig_vec / np.sum(max_eig_vec)
    
    # 步骤3：一致性检验
    n = judgment_matrix.shape[0]
    # 计算一致性指标CI
    ci = (max_eig_val - n) / (n - 1)
    # 随机一致性指标RI（n=2时RI=0，因为二阶矩阵天然一致）
    ri = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12}[n]
    # 一致性比率CR（CR<0.1则通过检验）
    cr = ci / ri if ri != 0 else 0
    
    if cr < 0.1:
        print(f"AHP一致性检验通过（CR={cr:.4f}<0.1）")
        print(f"成本权重：{weights[0]:.4f}，耗时权重：{weights[1]:.4f}")
        return weights
    else:
        raise ValueError(f"AHP一致性检验失败（CR={cr:.4f}≥0.1），请调整判断矩阵！")

# ===================== 2. 多目标整数规划问题定义 =====================
class CostTimeOptimization(ElementwiseProblem):
    def __init__(self):
        # 定义决策变量：1个整数变量（火箭发射总次数x）
        vars_num = 1  # 决策变量个数
        xl = np.array([0])  # 修正：必须是数组（维度匹配）
        xu = np.array([800000])  # 修正：必须是数组
        
        # 初始化问题：2个目标（最小化成本、最小化耗时），无约束（n_constr=0）
        super().__init__(
            n_var=vars_num,  # 决策变量数
            n_obj=2,        # 目标函数数
            n_constr=0,     # 无约束条件（若有约束需修改）
            xl=xl,          # 变量下限（数组）
            xu=xu,          # 变量上限（数组）
            type_var=int    # 变量类型：整数
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        定义目标函数（核心修正）
        :param x: 决策变量数组，如[x1]（x1=火箭发射总次数）
        :param out: 输出字典，F=[成本, 耗时]（匹配后续处理顺序）
        """
        x1 = x[0]  # 修正：取数组标量值（火箭发射总次数）
        
        # -------- 目标函数（保持你的业务逻辑） --------
        # 太空天梯运输时间
        time1 = (M - x1 * CAPACITY_2) / (CAPACITY_1 * N_1)      
        # 火箭运输时间
        time2 = x1 / I                                           
        # 总运输时间（年）
        total_time = max(time1, time2)                        
        # 太空天梯运输成本
        cost1 = N_1 * time1 * PRICE_1_1 + PRICE_1_2 * math.ceil((M - x1 * 125) / 125)
        # 火箭运输成本
        cost2 = x1 * PRICE_2
        # 总成本
        total_cost = cost1 + cost2
        
        # 修正1：用out["F"]输出目标值；修正2：顺序为[成本, 耗时]（匹配后续处理）
        out["F"] = [total_cost, total_time]

# ===================== 3. NSGA-II求解 + 综合得分选最优 =====================
def main():
    # ---------------- 步骤1：AHP计算权重 ----------------
    # 构建判断矩阵（成本和耗时同等重要）
    # 行/列：[成本, 耗时]
    judgment_matrix = np.array([
        [1, CVST],   # 修正注释：成本 vs 耗时=1（同等重要）
        [1/CVST, 1]    # 耗时 vs 成本=1
    ])
    weights = ahp_calculate_weight(judgment_matrix)
    
    # ---------------- 步骤2：定义优化问题 ----------------
    problem = CostTimeOptimization()
    
    # ---------------- 步骤3：配置NSGA-II算法 ----------------
    algorithm = NSGA2(
        pop_size=POP_SIZE,  # 种群规模
        sampling=IntegerRandomSampling(),  # 整数随机采样
        crossover=SBX(prob=0.9, eta=15, vtype=int),  # 整数交叉算子
        mutation=PM(prob=1/1, eta=20, vtype=int),    # 修正：变异概率（1/n_var更合理）
        eliminate_duplicates=True  # 去重
    )
    
    # ---------------- 步骤4：运行优化 ----------------
    res = minimize(
        problem,
        algorithm,
        ("n_gen", GEN_T),  # 迭代次数
        seed=SEED,        # 随机种子（可复现）
        verbose=True     # 打印迭代过程
    )
    
    # ---------------- 步骤5：处理帕累托解 ----------------
    # res.F 每行：[成本, 耗时]（匹配_evaluate的输出顺序）
    pareto_cost = res.F[:, 0]    # 成本（正确）
    pareto_time = res.F[:, 1]    # 耗时（正确）
    pareto_solutions = res.X     # 决策变量（火箭发射次数）
    df = pd.DataFrame(columns=["X", "cost", "norm_cost", "time" ,"norm_time", "score"])
    
    # 标准化目标值（消除量纲差异）
    # 处理极端情况：若所有值相同，避免除以0
    # cost_min, cost_max = np.min(pareto_cost), np.max(pareto_cost)
    # time_min, time_max = np.min(pareto_time), np.max(pareto_time)
    cost_min, cost_max = 44200000000000, 48000000000000
    time_min, time_max = 102.986731843575, 230.4147465
    
    norm_cost = (pareto_cost - cost_min) / (cost_max - cost_min) if cost_max != cost_min else 0
    norm_time = (pareto_time - time_min) / (time_max - time_min) if time_max != time_min else 0
    
    # 计算综合得分（权重×标准化值，得分越低越优）
    total_score = weights[0] * norm_cost + weights[1] * norm_time
    
    df["X"] = [int(x[0]) for x in pareto_solutions]
    df["cost"] = pareto_cost
    df["norm_cost"] = norm_cost
    df["time"] = pareto_time
    df["norm_time"] = norm_time
    df["score"] = total_score
    
    
    # 找到综合得分最低的最优方案
    best_idx = np.argmin(total_score)
    best_solution = pareto_solutions[best_idx]
    best_cost = pareto_cost[best_idx]
    best_time = pareto_time[best_idx]
    best_score = total_score[best_idx]
    
    df["best_solution?"] = [1 if i == best_idx else 0 for i in range(len(pareto_solutions))]
    df.to_csv("Pareto_Scenarios_New_Solution.csv", index = False)
    
    # ---------------- 步骤6：输出结果 ----------------
    print("\n================= 最优方案结果 =================")
    print(f"最优决策变量：火箭发射总次数={int(best_solution[0])}")
    print(f"最优方案成本：{best_cost:.2e}")
    print(f"最优方案耗时：{best_time:.2f} 年")
    print(f"最优方案综合得分：{best_score:.4f}（最低）")
    
    # ---------------- 步骤7：可视化帕累托前沿 ----------------
    plt.figure(figsize=(8, 6))
    # 绘制所有帕累托解
    plt.scatter(pareto_cost/1e9, pareto_time, label="帕累托最优解", color="blue", alpha=0.6)
    # 标记最优方案（成本转1e9单位，提升图表可读性）
    plt.scatter(best_cost/1e9, best_time, label="综合得分最优解", color="red", s=100, marker="*")
    plt.xlabel("成本（×10^9 元，越小越好）")
    plt.ylabel("耗时（年，越小越好）")
    plt.title("成本-耗时帕累托前沿 + 最优方案")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()