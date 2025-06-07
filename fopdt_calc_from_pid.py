import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import time
import pandas as pd

df = pd.read_csv('combined_output.csv')
t_array = df['Timestamp'].to_numpy()
pv_array = df['PV'].to_numpy()
cv_array = df['CV'].to_numpy()
sp_array = df['SP'].to_numpy()

cv_min = 0
cv_max = 100
pv_min = 0
pv_max = 250
kp = 0.173
ki = 0.702
kd = -0.004
dead_time = 0.1
tangent_time = 0.75
process_gain = 0.875
time_constant = max(1.1 * (tangent_time - dead_time), 0.1)
pv_init = 40

sim_time = 20

timestep = t_array[1] - t_array[0]

# Live Plotting
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
(pv_line,) = ax.plot([], [], label='PV')
(pv_sim_line,) = ax.plot([], [], label='PV Sim')
(cv_line,) = ax.plot([], [], label='CV')
ax.set_xlim(0, sim_time)
ax.set_ylim(0, max(cv_max,pv_max))
ax.set_xlabel("Time (s)")
ax.set_ylabel("Values")
ax.set_title("PID Optimization")
ax.legend()
fig.tight_layout()

# System Simulation
def simulate_system(t_array, cv_array, process_gain, time_constant, dead_time, pv_init):
    dt = np.mean(np.diff(t_array))
    
    n_steps = len(t_array)
    pv_array = np.zeros(n_steps)
    pv_array[0] = pv_init
    dpv = pv_init
    delay_steps = max(1, int(np.round(dead_time / dt)))
    cv_buffer = [cv_array[0]] * delay_steps

    # Run the actual simulation for x time
    for i in range(1, n_steps):
        cv_buffer.append(cv_array[i])
        delayed_cv = cv_buffer.pop(0)
        
        dpv = (-pv_array[i-1] + process_gain * delayed_cv) / time_constant
        pv_array[i] = pv_array[i-1] + dpv *dt

    return np.array(pv_array)

trial_counter = 0
def cost_function(fopdt_params):
    global trial_counter
    process_gain, time_constant, dead_time = fopdt_params
    
    pv = simulate_system(t_array, cv_array, process_gain, time_constant, dead_time, pv_init)
    
    if np.any(np.isnan(pv)) or np.any(np.isinf(pv)):
        return 1e6
    
    # Live plotting
    pv_sim_line.set_data(t_array, pv)
    pv_line.set_data(t_array, pv_array)
    cv_line.set_data(t_array, cv_array)
    ax.set_ylim(min(pv.min(), pv_array.min()) - 2, max(pv.max(), pv_array.max()) + 2)
    ax.set_title(f"Trial #{trial_counter+1} | Gain={process_gain:.3f}, TC={time_constant:.3f}, DT={dead_time:.3f}")
    fig.canvas.draw()
    fig.canvas.flush_events()
    trial_counter += 1
    error = pv_array-pv
    
    ise = abs(np.sum(np.square(error)))
    # Overshoot Penalty Penalty calculation
    overshoot = np.maximum(0, abs(pv_array-pv))
    overshoot_penalty = np.sum(np.square(overshoot))
    alpha = 1.0
    beta = 1.0
    cost = (ise * alpha) + (overshoot_penalty * beta)
    return cost

#pv_sim = simulate_system(t_array, cv_array, process_gain, time_constant, dead_time, pv_init=40)
bounds = [(0,50), (0,50), (0,50)]
#result = differential_evolution(cost_function, bounds, seed=42, strategy='best1bin', maxiter=100, popsize=15, tol=1e-6)
#gain_opt, tc_opt, dt_opt = result.x
gain_opt = process_gain
tc_opt = time_constant
dt_opt = dead_time
pv_sim = simulate_system(t_array, cv_array, gain_opt, tc_opt, dt_opt, pv_init=40)
print(f"\n✅ Optimized PID gains: Gain={gain_opt:.3f}, TC={tc_opt:.3f}, DT={dt_opt:.3f}")
plt.figure(figsize=(10, 5))
plt.plot(t_array, pv_array, label='PV Actual')
plt.plot(t_array, pv_sim, label='PV Simulated')
plt.plot(t_array, cv_array, label='CV')
plt.xlabel("Time (s)")
plt.ylabel("Values")
plt.title("Final Optimized PID Response")
plt.legend()
plt.grid(True)
plt.tight_layout()

#print(cv_buffer[0:50])

plt.show()