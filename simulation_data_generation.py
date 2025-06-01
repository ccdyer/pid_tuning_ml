import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import time
import pandas as pd

starttime = time.time()
pv_init = 8.0
pv_min = 0
pv_max = 250
cv_init = 10.0
cv_steps = [[10.0,0],[30,5.0]]

dead_time = 0.1
tangent_time = 0.75
process_gain = 0.875
time_constant = max(1.1 * (tangent_time - dead_time), 0.1)

points_per_second = 50
if dead_time <=1:
    sim_time = 20
    num_points = 1000
else:
    sim_time = int(dead_time * 20)
    num_points = int(sim_time * points_per_second)
    
# Create an array of numbers over the interval for the time scale
# Start at zero, end at sim_time, create num_points number of samples
t_eval = np.linspace(0, sim_time, num_points)

# Calculate the time step
# In default case, this will be 20 seconds/1000 steps=20mSec
timestep = t_eval[1] - t_eval[0]

scenario= {
    "disturbance": False,
    "measurement_noise": False
}

# System Simulation
def simulate_system(cv_array):
    # Initialize lists for cv/pv/sp histories
    cv_history = []
    pv_array = []
    t_array = []
    
    # Initialize CV and PV based on setpoints
    pv = np.clip(pv_init, pv_min, pv_max)
    # Fill out the cv buffer with the CV init value to avoid starting at zero
    cv_delay_steps = max(1, int(np.round(dead_time / timestep)))
    cv_buffer = [cv_init] * cv_delay_steps

    # Run the actual simulation for x time
    for i, t in enumerate(t_eval):
        # Update the PID every cycle
        for x in cv_array:
            if t >= x[1]:
                cv = x[0]
        
        cv_buffer.append(cv)
        delayed_cv = cv_buffer.pop(0)
        # Update the PID simulation
        dpv = (-pv + process_gain * delayed_cv) / time_constant
        pv += dpv * timestep
        pv = np.clip(pv, pv_min, pv_max)

        # Measurement Noise logic
        if scenario["measurement_noise"]:
            pv += np.random.normal(0, measurement_noise_magnitude)
        
        # Add the PV, cv and SP to arrays for plotting
        pv_array.append(pv)
        cv_history.append(cv)
        t_array.append(t)
        

    return np.array(pv_array), np.array(cv_history), np.array(t_array)

pv, cv, t = simulate_system(cv_steps)

combined = np.column_stack((t, pv, cv))
headers = ["Timestamp", "PV", "CV"]
df = pd.DataFrame(combined, columns=headers)
df.to_csv('combined_output.csv', index=False)


plt.figure(figsize=(10, 5))
plt.plot(t, pv, label='PV')
plt.plot(t, cv, 'g:', label='CV')
plt.xlabel("Time (s)")
plt.ylabel("Values")
plt.title("Final Optimized PID Response")
plt.legend()
plt.grid(True)
plt.tight_layout()

endtime = time.time()
print(f"Elapsed Time: {(endtime - starttime) / 60:.2f} minutes")
#print(cv_buffer[0:50])

plt.show()