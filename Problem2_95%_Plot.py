import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl
import warnings

# Configuration
warnings.filterwarnings('ignore')
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

# Parameters
M_TOTAL = 100000000
N_PORTS = 3
CAPACITY_ELEVATOR = 179000
CAPACITY_ROCKET = 125
FREQ_ROCKET_MAX = 3472

# Cost Parameters (Mean, Std)
COST_EARTH_ROCKET_MEAN = 60000000
COST_EARTH_ROCKET_STD = 2500000
COST_ELEV_ROCKET_MEAN = 54000000
COST_ELEV_ROCKET_STD = 2250000
COST_ELEVATOR_OPS_MEAN = 1790000000
COST_ELEVATOR_OPS_STD = 5000000

N_SIM = 10000
SEED = 725

# ================= Data Generation =================
print("1. Initializing noise matrix...")
np.random.seed(SEED)
fixed_cost_earth_rocket = np.random.normal(COST_EARTH_ROCKET_MEAN, COST_EARTH_ROCKET_STD, N_SIM)
fixed_cost_elev_rocket = np.random.normal(COST_ELEV_ROCKET_MEAN, COST_ELEV_ROCKET_STD, N_SIM)
fixed_cost_elevator_ops = np.random.normal(COST_ELEVATOR_OPS_MEAN, COST_ELEVATOR_OPS_STD, N_SIM)
fixed_fail_rocket = np.random.binomial(1, 0.03, N_SIM)
fixed_fail_elevator = np.random.binomial(1, 0.02, N_SIM)

r_eff_factor = np.where(fixed_fail_rocket == 1, 0.5, 1.0)
r_repair_time = np.where(fixed_fail_rocket == 1, 14/365, 0.0)
e_eff_factor = np.where(fixed_fail_elevator == 1, 0.8, 1.0)
e_repair_time = np.where(fixed_fail_elevator == 1, 0.1, 0.0)
total_elevator_cap_array = N_PORTS * CAPACITY_ELEVATOR * e_eff_factor

def scan_robust_bands():
    print("2. Scanning domain (Calculating Confidence Intervals)...")
    x_values = np.arange(0, 800001, 2000) 
    
    results = {
        'x': x_values,
        'cost_mean': [], 'cost_05': [], 'cost_25': [], 'cost_75': [], 'cost_95': [],
        'time_mean': [], 'time_05': [], 'time_25': [], 'time_75': [], 'time_95': []
    }
    
    batch_size = 100
    total_batches = len(x_values) // batch_size + 1
    
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(x_values))
        if start_idx >= end_idx: break
        current_x = x_values[start_idx:end_idx][:, np.newaxis]
        
        # Physical Calculation
        m_earth = current_x * CAPACITY_ROCKET
        m_elev = np.maximum(0, M_TOTAL - m_earth)
        n_elev_launch = m_elev / CAPACITY_ROCKET
        
        # Time Calculation
        t_r = (current_x / (FREQ_ROCKET_MAX * r_eff_factor)) + r_repair_time
        with np.errstate(divide='ignore', invalid='ignore'):
            t_e = (m_elev / total_elevator_cap_array) + e_repair_time
            t_e = np.where(m_elev <= 0, 0, t_e)
        final_time = np.maximum(t_r, t_e)
        
        # Cost Calculation
        c_tot = (current_x * fixed_cost_earth_rocket) + \
                (n_elev_launch * fixed_cost_elev_rocket) + \
                (N_PORTS * final_time * fixed_cost_elevator_ops)
        
        # Statistics
        results['cost_mean'].extend(np.mean(c_tot, axis=1))
        results['cost_05'].extend(np.quantile(c_tot, 0.05, axis=1))
        results['cost_25'].extend(np.quantile(c_tot, 0.25, axis=1))
        results['cost_75'].extend(np.quantile(c_tot, 0.75, axis=1))
        results['cost_95'].extend(np.quantile(c_tot, 0.95, axis=1))
        
        results['time_mean'].extend(np.mean(final_time, axis=1))
        results['time_05'].extend(np.quantile(final_time, 0.05, axis=1))
        results['time_25'].extend(np.quantile(final_time, 0.25, axis=1))
        results['time_75'].extend(np.quantile(final_time, 0.75, axis=1))
        results['time_95'].extend(np.quantile(final_time, 0.95, axis=1))
        
    return {k: np.array(v) for k, v in results.items()}

# ================= Plotting Function =================
def plot_high_end_robustness(data):
    COLOR_COST = "#32bbcd"
    COLOR_TIME = "#9368ab"

    fig, ax1 = plt.subplots(figsize=(12, 7))
    x = data['x']
    ax1.grid(True, which='major', color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    # === Plot Cost (Left Axis) ===
    ax1.plot(x, data['cost_mean'], color=COLOR_COST, linewidth=2, label='Expected Cost')
    ax1.fill_between(x, data['cost_05'], data['cost_95'], color=COLOR_COST, alpha=0.15, 
                     label='90% Cost CI (Volatility)')
    ax1.fill_between(x, data['cost_25'], data['cost_75'], color=COLOR_COST, alpha=0.3)

    # === Plot Time (Right Axis) ===
    ax2 = ax1.twinx()
    ax2.plot(x, data['time_mean'], color=COLOR_TIME, linewidth=2, linestyle='--', label='Expected Time')
    ax2.fill_between(x, data['time_05'], data['time_95'], color=COLOR_TIME, alpha=0.3, 
                     label='90% Time CI (Uncertainty)')
    ax2.fill_between(x, data['time_25'], data['time_75'], color=COLOR_TIME, alpha=0.6)

    # === Highlight Optimal Point ===
    idx_opt = np.argmin(data['time_95'])
    x_opt = x[idx_opt]
    y_opt_time = data['time_95'][idx_opt]
    
    ax2.scatter(x_opt, y_opt_time, color=COLOR_TIME, s=300, zorder=10, 
                edgecolors='white', linewidth=1.5, marker='*')
    
    ax2.text(x_opt+50000, y_opt_time - 2, 
             '(357569, 102.99)', 
             color=COLOR_TIME, 
             fontsize=11, 
             fontweight='bold', 
             ha='center', 
             va='bottom')

    # === Axis Formatting ===
    ax1.set_xlabel('Rocket Launches', fontsize=16, fontweight='bold')
    
    ax1.set_ylabel('Total Cost (USD)', fontsize=16, color=COLOR_COST, fontweight='bold')
    ax1.tick_params(axis='y', colors=COLOR_COST)
    ax1.spines['left'].set_color(COLOR_COST)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['right'].set_visible(False)

    ax2.set_ylabel('Project Duration (Years)', fontsize=16, color=COLOR_TIME, fontweight='bold')
    ax2.tick_params(axis='y', colors=COLOR_TIME)
    ax2.spines['right'].set_color(COLOR_TIME)
    ax2.spines['right'].set_linewidth(1.5)
    ax2.spines['left'].set_visible(False)

    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f'${x/1e13:.1f}T'))
    ax1.set_xlim(0, 800000)
    
    # === Legend ===
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    from matplotlib.patches import Patch
    patch1 = Patch(facecolor=COLOR_COST, alpha=0.3, label='Cost Variance (50% & 90% CI)')
    patch2 = Patch(facecolor=COLOR_TIME, alpha=0.2, label='Time Variance (50% & 90% CI)')
    
    legend = ax1.legend(lines1 + [patch1] + lines2 + [patch2], 
               [l for l in labels1] + ['Cost Uncertainty Range'] + [l for l in labels2] + ['Time Uncertainty Range'],
               loc='lower center',
               bbox_to_anchor=(0.3, 0.8),
               ncol=2, 
               frameon=True, 
               fancybox=True, 
               shadow=True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = scan_robust_bands()
    plot_high_end_robustness(data)