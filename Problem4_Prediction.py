import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# Constants
START_YEAR, END_YEAR = 2025, 2050
N_0 = 42529 
SOLAR_PERIOD = 11.0

def debris_model(N, t, params):
    source = params['L_rate'] + params['E_rate'] + params['alpha'] * (N**2)
    solar_factor = 1 + params['solar_amp'] * np.cos(2 * np.pi * t / SOLAR_PERIOD)
    sink = params['decay_base'] * solar_factor * N
    return source - sink

t_span = np.linspace(0, END_YEAR - START_YEAR, (END_YEAR - START_YEAR)*12 + 1)
years = START_YEAR + t_span

# Scenarios
scenarios = {
    'Nominal': {'L_rate': 2500, 'E_rate': 135, 'alpha': 2.5e-7, 'decay_base': 0.025, 'solar_amp': 0.4},
    'Pessimistic': {'L_rate': 3000, 'E_rate': 180, 'alpha': 3.0e-7, 'decay_base': 0.022, 'solar_amp': 0.35},
    'Optimistic': {'L_rate': 2000, 'E_rate': 80, 'alpha': 2.0e-7, 'decay_base': 0.028, 'solar_amp': 0.45}
}

results = {name: odeint(debris_model, N_0, t_span, args=(p,)).flatten() for name, p in scenarios.items()}
pred_nom = results['Nominal'][-1]

# Visualization
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=120)
ax2 = ax1.twinx()

# Solar Cycle Background
ax2.plot(years, 1 + 0.4 * np.cos(2 * np.pi * t_span / SOLAR_PERIOD), color='orange', linestyle=':', label='Solar Activity Index')
ax2.set_ylabel('Solar Activity Index', color='orange', fontweight='bold')
ax2.tick_params(axis='y', colors='orange')
ax2.spines['right'].set_color('orange')

# Debris Plot
ax1.fill_between(years, results['Optimistic'], results['Pessimistic'], color="#b48fbbff", alpha=0.4, label='Uncertainty Interval (95% CI)')
ax1.plot(years, results['Nominal'], color="#7d4e86", linewidth=2.5, label='Nominal Prediction')
ax1.scatter([2050], [pred_nom], color="#EF1616", s=100, zorder=5)

ax1.annotate(f'2050 Projection:\nNominal: {int(pred_nom):,}', xy=(2050, pred_nom), xytext=(2035, pred_nom + 8000),
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9),
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color='#333'))

ax1.set_title('Projected Evolution of Space Debris Population (>10cm)', fontweight='bold')
ax1.set_xlabel('Year', fontweight='bold')
ax1.set_ylabel('Number of Objects', color="#7d4e86", fontweight='bold')
ax1.tick_params(axis='y', colors="#7d4e86")
ax1.spines['left'].set_color("#7d4e86")
ax1.set_xlim(2025, 2050)
ax1.grid(True, linestyle='--', alpha=0.6)

# Legends
l1, lab1 = ax1.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lab1 + lab2, loc='upper left')

plt.tight_layout()
plt.show()