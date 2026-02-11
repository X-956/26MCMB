import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
import matplotlib as mpl
import warnings

# Configuration
warnings.filterwarnings('ignore')
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

class Config:
    M = 100000000
    C1, C2 = 179000, 125
    N1 = 3
    n2_annual = 3472
    P11, P12 = 1790000000, 0.9 * 60000000
    
    Price_Normal, Env_Factor_Normal = 60000000, 1.0 
    Price_Green, Env_Factor_Green = 120000000, 0.4 
    
    Propellant = 549_000
    Base_Emission = 0.914 + 0.022
    Debris_Base = 94_802

class SpaceLogisticsProblem(ElementwiseProblem):
    def __init__(self):
        self.max_N2 = int(Config.M / Config.C2)
        # x[0]: N2 (Total Launches), x[1]: Green_Ratio
        super().__init__(n_var=2, n_obj=3, n_constr=0, 
                         xl=np.array([0, 0]), xu=np.array([self.max_N2, 1]))

    def _evaluate(self, x, out, *args, **kwargs):
        N2 = int(np.round(x[0])) 
        ratio = x[1]
        
        y = N2 * Config.C2
        remain_x = max(0, Config.M - y)
        
        # 1. Time
        T1 = remain_x / (Config.N1 * Config.C1) if remain_x > 0 else 0
        T2 = N2 / Config.n2_annual if N2 > 0 else 0
        Time = max(T1, T2)
        
        # 2. Cost
        N_green = N2 * ratio
        N_normal = N2 * (1 - ratio)
        Cost_Elevator = (Config.N1 * T1 * Config.P11) + (np.ceil(remain_x/Config.C2) * Config.P12)
        Cost_Rocket = (N_normal * Config.Price_Normal) + (N_green * Config.Price_Green)
        Total_Cost = Cost_Elevator + Cost_Rocket
        
        # 3. Environment
        Gas = (N_normal * Config.Env_Factor_Normal + N_green * Config.Env_Factor_Green) * Config.Propellant * Config.Base_Emission
        Debris = (2 * N_normal + 2 * N_green * Config.Env_Factor_Green) / 0.17
        Env_Score = Gas + (Config.Debris_Base + Debris) * 1000
        
        out["F"] = [Time, Total_Cost/1e9, Env_Score]

def solve_with_nsga3():
    print("Initializing NSGA-III...")
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=30)
    algorithm = NSGA3(pop_size=500, ref_dirs=ref_dirs, sampling=FloatRandomSampling(),
                      crossover=SBX(prob=0.8, eta=10), mutation=PM(eta=15), eliminate_duplicates=True)
    res = minimize(SpaceLogisticsProblem(), algorithm, get_termination("n_gen", 100), seed=100, verbose=True)
    
    data = np.hstack([res.F, res.X])
    df = pd.DataFrame(data, columns=['Time', 'Cost', 'Env', 'N2', 'Green_Ratio'])
    df['N2'] = df['N2'].astype(int)
    return df

# Main Logic
df = solve_with_nsga3()
env_min, env_max = df['Env'].min(), df['Env'].max()
df['Env_Norm'] = (df['Env'] - env_min) / (env_max - env_min)

# Find Best Point (AHP Weights: Time=1, Cost=2, Env=3)
def normalize(s): return (s - s.min()) / (s.max() - s.min())
df['Score'] = normalize(df['Time'])*1 + normalize(df['Cost'])*2 + df['Env_Norm']*3
best_pt = df.loc[df['Score'].idxmin()]

print(f"\nBest AHP Score Point: Time={best_pt['Time']:.2f}, Cost={best_pt['Cost']:.2f}, Env={best_pt['Env_Norm']:.4f}")

# Plotting Configuration
colors = ["#ec0400", "#f2724d", "#fee395", "#fff9ab", "#afdef4", "#298df0"]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df['Time'], df['Cost'], df['Env_Norm'], c=df['Green_Ratio'], cmap=cmap, s=28, alpha=0.75)
ax.set_xlabel('Year')
ax.set_ylabel('Cost (USD)')
ax.set_zlabel('Env Impact (Normalized)')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1e3:.1f}T'))
plt.colorbar(sc, label='Green Rocket Ratio', shrink=0.6)
plt.title('Pareto Front via NSGA-III')
plt.show()