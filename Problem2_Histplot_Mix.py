import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import warnings

# Configuration
warnings.filterwarnings('ignore')
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

# Constants
M_TOTAL = 100000000
N_PORTS = 3
CAPACITY_ELEVATOR = 179000
CAPACITY_ROCKET = 125
FREQ_ROCKET_MAX = 3472

# Decision Variables (Mixed Scheme)
X_LAUNCHES = 357569                                          
M_ELEVATOR = M_TOTAL - X_LAUNCHES * CAPACITY_ROCKET

# Monte Carlo Parameters
N_SIM = 10000
SEED = 2024

def run_monte_carlo():
    np.random.seed(SEED)
    print(f"--- Monte Carlo Simulation (N={N_SIM}) ---")
    print(f"Scheme: {X_LAUNCHES} Rocket Launches, {M_ELEVATOR/1e6:.2f}M tons via Elevator")

    # 1. Generate Random Samples
    sim_cost_rocket_unit = np.random.normal(60000000, 2500000, N_SIM)
    sim_cost_elevator_year = np.random.normal(1790000000, 5000000, N_SIM)
    sim_fail_rocket = np.random.binomial(1, 0.03, N_SIM)
    sim_fail_elevator = np.random.binomial(1, 0.01, N_SIM)

    total_costs = []
    total_times = []

    # 2. Simulation Loop
    for i in range(N_SIM):
        # Time Calculation with Failures
        r_eff = 0.5 if sim_fail_rocket[i] == 1 else 1.0
        r_repair = (14 / 365) if sim_fail_rocket[i] == 1 else 0
        time_rocket = (X_LAUNCHES / (FREQ_ROCKET_MAX * r_eff)) + r_repair
        
        e_eff = 0.8 if sim_fail_elevator[i] == 1 else 1.0
        e_repair = 0.1 if sim_fail_elevator[i] == 1 else 0
        total_elevator_capacity = N_PORTS * CAPACITY_ELEVATOR * e_eff
        time_elevator = (M_ELEVATOR / total_elevator_capacity) + e_repair
        
        c_ele_ro = math.ceil(M_ELEVATOR / 125) * sim_cost_rocket_unit[i]
        final_time = max(time_rocket, time_elevator)
        total_times.append(final_time)
        
        # Cost Calculation
        c_rocket_total = X_LAUNCHES * sim_cost_rocket_unit[i]
        c_elevator_total = N_PORTS * final_time * sim_cost_elevator_year[i] + c_ele_ro
        total_costs.append(c_rocket_total + c_elevator_total)

    total_costs = np.array(total_costs)
    total_times = np.array(total_times)

    # 3. Statistics
    mean_cost = np.mean(total_costs)
    var_cost_95 = np.percentile(total_costs, 95)
    mean_time = np.mean(total_times)
    var_time_95 = np.percentile(total_times, 95)
    
    print("\n=== Simulation Statistics ===")
    print(f"Mean Cost: ${mean_cost/1e9:.2f} Billion")
    print(f"Cost VaR 95%: ${var_cost_95/1e9:.2f} Billion")
    print(f"Risk Premium: {(var_cost_95 - mean_cost)/mean_cost*100:.2f}%")
    print(f"Mean Time: {mean_time:.2f} Years")
    print(f"Time VaR 95%: {var_time_95:.2f} Years")
    
    # 4. Visualization (Histogram)
    plt.figure(figsize=(6, 4))
    
    # Cost Distribution
    sns.histplot(total_costs/1e12, kde=True, element='bars', edgecolor="#4d734b", linewidth=0.7, color="#85b779a7", bins=50, alpha = 0.5)
    plt.axvline(mean_cost/1e12, color="#ffb85bff", linestyle='-.', label=f'Mean: {mean_cost/1e12:.1f} T')
    plt.axvline(var_cost_95/1e12, color="#277e8d", linestyle='--', label=f'VaR 95%: {var_cost_95/1e12:.1f} T')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, p:f'{x:.1F}T'))
    plt.xlim(41,58)
    plt.title('Total Cost Distribution (Mixed Scheme)')
    plt.xlabel('Cost (USD)')
    plt.legend()
    plt.savefig('Monte_Carlo_Simulation(Mix).png', dpi=900, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_monte_carlo()