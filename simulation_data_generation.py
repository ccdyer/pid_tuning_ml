import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import time
import pandas as pd

starttime = time.time()

# Simulation Parameters
cv_start = 30.0
cv_final = 70.0
cv_min = 0
cv_max = 100
cv_init = 56.74
pv_start = 10.0
pv_final = 45.0
pv_min = 0
pv_max = 250
pv_init = 50.02
process_gain = 0.85
dead_time = 0.09
tangent_time = 0.73
setpoint = 50
setpoint_2 = 62
setpoint_2_time = 10.0
settling_tolerance = 0.005
manual_kp = 0.185
manual_ki = 0.7
manual_kd = 0.0

# Calculated Parameters
cv_step_change = cv_final - cv_start
pv_change = pv_final - pv_start
#process_gain = pv_change / cv_step_change
time_constant = max(1.1 * (tangent_time - dead_time), 0.1)

# Calculate sim time and number of points based on dead time
# If dead time is more than 5%(ish) length of simulation, the simulation breaks
# Currently using 50 points per second, seems to work well
# And be fast enough for most applications
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

# Scenario flags
# Change scenarios to True if you want to test using them
# disturbance tests a disturbance of set magnitude at set time
# measurement noise tests with random measurement noise inserted onto the PV
# magnitude is in decimal, 0.1=10%
scenario= {
    "disturbance": False,
    "measurement_noise": True
}
# disturbance_magnitude = 0.01
# disturbance_time = 10
# disturbance_duration = 2.0
# disturbance_end = disturbance_time + disturbance_duration
# disturbance_value = setpoint * disturbance_magnitude
measurement_noise_magnitude = 0.1
#print(disturbance_value)

# PID Controller Class
class PID:
    # Initialize PID, set parameters and set integral and error to zero
    def __init__(self, Kp, Ki, Kd, cv_init):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral = cv_init / self.Ki if self.Ki > 0 else 0.0
        self.prev_error = 0

    # Reset PID, set integral and error to zero
    def reset(self):
        self.integral = cv_init / self.Ki if self.Ki > 0 else 0.0
        self.prev_error = 0
        
    # Core PID logic  
    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        control = np.clip(self.Kp * error + self.Ki * self.integral + self.Kd * derivative, cv_min, cv_max)
        self.prev_error = error
        integral = self.integral
        return control, integral

# System Simulation
def simulate_system(Kp, Ki, Kd):
    # Call the PID and make sure it's reset at the start
    pid = PID(Kp, Ki, Kd, cv_init)
    pid.reset()

    # Initialize lists for cv/pv/sp histories
    cv_history = []
    pv_array = []
    sp_array = []
    
    # Initialize CV and PV based on setpoints
    pv = np.clip(pv_init, pv_min, pv_max)
    # Fill out the cv buffer with the CV init value to avoid starting at zero
    cv_delay_steps = max(1, int(np.round(dead_time / timestep)))
    cv_buffer = [cv_init] * cv_delay_steps
    
    disturbance_magnitude = 0.001
    disturbance_time = 5
    disturbance_duration = 3.0
    disturbance_end = disturbance_time + disturbance_duration
    disturbance_value = setpoint * disturbance_magnitude
    
    #sp = setpoint
    # Run the actual simulation for x time
    for i, t in enumerate(t_eval):
        if t >= setpoint_2_time:
            sp = setpoint_2
            #print(sp)
        else:
            sp = setpoint
        #print(sp)
        # Update the PID every cycle
#         if scenario["disturbance"] and disturbance_time <= t < disturbance_end:
#             sp = sp - disturbance_value
#         else:
#             sp = setpoint
        error = sp - pv
        cv, integral = pid.update(error, timestep)
        
        cv_buffer.append(cv)
        delayed_cv = cv_buffer.pop(0)
        # Update the PID simulation
        dpv = (-pv + process_gain * delayed_cv) / time_constant
        pv += dpv * timestep
        pv = np.clip(pv, pv_min, pv_max)

        # Measurement Noise logic
        if scenario["measurement_noise"]:
            pv += np.random.normal(0, measurement_noise_magnitude)
        #print(sp)
        # Add the PV, cv and SP to arrays for plotting
        pv_array.append(pv)
        cv_history.append(cv)
        sp_array.append(sp)

    return np.array(pv_array), np.array(sp_array), np.array(cv_history)

# PID bounds, set to values based on process
# If it recommends max value I recommend increasing
# D stands for do not use(unless you need it)

kp_opt = manual_kp
ki_opt = manual_ki
kd_opt = manual_kd
# --- Final Plot ---
plt.ioff()
pv, sp, cv= simulate_system(kp_opt, ki_opt, kd_opt)


overshoot = pv - sp
overshoot[overshoot < 0] = 0
max_overshoot = np.max(overshoot)
max_overshoot_pct = (max_overshoot/(pv_max-pv_min))*100
print(f"📈 Maximum overshoot: {max_overshoot:.3f}/{max_overshoot_pct:.3f}%")

plt.figure(figsize=(10, 5))
plt.plot(t_eval, pv, label='PV')
plt.plot(t_eval, sp, 'r--', label='Setpoint')
plt.plot(t_eval, cv, 'g:', label='CV')
plt.xlabel("Time (s)")
plt.ylabel("Values")
plt.title("Final Optimized PID Response")
plt.legend()
plt.grid(True)
plt.tight_layout()

combined = np.column_stack((t_eval, pv, cv, sp))
headers = ["Timestamp", "PV", "CV", "SP"]
df = pd.DataFrame(combined, columns=headers)
df.to_csv('combined_output2.csv', index=False)

endtime = time.time()
print(f"Elapsed Time: {(endtime - starttime) / 60:.2f} minutes")
#print(cv_buffer[0:50])

plt.show()

