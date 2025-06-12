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
cv_init = 30
pv_start = 10.0
pv_final = 45.0
pv_min = 0
pv_max = 250
pv_init= 10
dead_time = 0.1
tangent_time = 0.75
setpoint = 45
settling_tolerance = 0.005
manual_kp = 0.185
manual_ki = 0.7
manual_kd = 0.0
manual_test = False

# Calculated Parameters
cv_step_change = cv_final - cv_start
pv_change = pv_final - pv_start
process_gain = pv_change / cv_step_change
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
    "measurement_noise": False
}
# disturbance_magnitude = 0.01
# disturbance_time = 10
# disturbance_duration = 2.0
# disturbance_end = disturbance_time + disturbance_duration
# disturbance_value = setpoint * disturbance_magnitude
measurement_noise_magnitude = 0.01
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
    sp = setpoint
    # Run the actual simulation for x time
    for i, t in enumerate(t_eval):
        # Update the PID every cycle
        if scenario["disturbance"] and disturbance_time <= t < disturbance_end:
            sp = sp - disturbance_value
        else:
            sp = setpoint
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
        
        # Add the PV, cv and SP to arrays for plotting
        pv_array.append(pv)
        cv_history.append(cv)
        sp_array.append(sp)

    return np.array(pv_array), np.array(sp_array), np.array(cv_history)

# Live Plotting
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
(pv_line,) = ax.plot([], [], label='PV')
(sp_line,) = ax.plot([], [], 'r--', label='Setpoint')
(cv_line,) = ax.plot([], [], 'g:', label='CV')
ax.set_xlim(0, sim_time)
ax.set_ylim(0, max(cv_max,pv_max))
ax.set_xlabel("Time (s)")
ax.set_ylabel("Values")
ax.set_title("PID Optimization")
ax.legend()
fig.tight_layout()

# Compute Settling Time
def compute_settling_time(pv, t, setpoint, tolerance=settling_tolerance):
    pv = np.asarray(pv)
    t = np.asarray(t)

    min_len = min(len(pv), len(t))
    pv = pv[:min_len]
    t = t[:min_len]

    band_upper = setpoint * (1 + tolerance)
    band_lower = setpoint * (1 - tolerance)

    for i in range(min_len):
        window = pv[i:]
        if np.all((window >= band_lower) & (window <= band_upper)):
            return t[i]
    return t[-1]  # If it never settles

# Cost Function, this is where the magic happens
trial_counter = 0
def cost_function(pid_gains):
    global trial_counter
    Kp, Ki, Kd = pid_gains
    
    # Run the system Simulation
    pv, sp, cv= simulate_system(Kp, Ki, Kd)

    # If PV is not a number then reject this solution
    if np.any(np.isnan(pv)) or np.any(np.isinf(pv)):
        return 1e6
    
    # Live plotting
    pv_line.set_data(t_eval, pv)
    sp_line.set_data(t_eval, sp)
    cv_line.set_data(t_eval, cv)
    ax.set_ylim(min(pv.min(), cv.min()) - 2, max(sp.max(), pv.max(), cv.max()) + 2)
    ax.set_title(f"Trial #{trial_counter+1} | Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}")
    fig.canvas.draw()
    fig.canvas.flush_events()
    trial_counter += 1

    # Calculate Error
    error = sp - pv
    
    # Weighting Calculations
    # Integrated Squared Error Calc
    # This is mainly used to avoid scenarios where the pv never reaches the setpoint
    ise = np.sum(np.square(error)) * timestep
    
    # Overshoot Penalty Penalty calculation
    overshoot = np.maximum(0, pv - sp)
    overshoot_penalty = np.sum(np.square(overshoot)) * timestep
    
    # Large initial PV change penalty
    # Penalize large PV step changes
    initial_window = int(1 / timestep)
    initial_pv = pv[0]
    transient_deviation = np.abs(pv[:initial_window] - initial_pv)
    initial_instability_penalty = np.sum(transient_deviation)
    
    # Large initial control actions penalty
    # Penalize large CV step changes
    initial_cv_penalty = np.sum(np.abs(np.diff(cv[:initial_window])))
    
    # Settling time Penalty
    settling_time = compute_settling_time(pv, t_eval, sp[-1])
    
    # Weighting Setpoints - adjust these based on your process to get the best response
    # ISE Weight
    alpha = 1.0
    # Overshoot Penalty Weight
    beta = 10.0
    # Large Initial PV Change Weight
    gamma = 1.0
    # Initial CV Jump Weight
    delta = 1.0
    # Settling Time Weight
    epsilon = 10.0
    
    # Total Cost Calc
    cost = (alpha * ise) + (beta * overshoot_penalty) + (gamma * initial_instability_penalty) + (delta * initial_cv_penalty) + (epsilon * settling_time) 
    return cost

# Optimization

# PID bounds, set to values based on process
# If it recommends max value I recommend increasing
# D stands for do not use(unless you need it)
if manual_test == False:
    bounds = [(0, 2), (0, 2), (0, 2)]  # You can adjust these
    result = differential_evolution(cost_function, bounds, seed=42, strategy='best1bin', maxiter=100, popsize=15, tol=1e-6)
    kp_opt, ki_opt, kd_opt = result.x
    print(f"\n✅ Optimized PID gains: Kp={kp_opt:.3f}, Ki={ki_opt:.3f}, Kd={kd_opt:.3f}")
else:
    kp_opt = manual_kp
    ki_opt = manual_ki
    kd_opt = manual_kd
# --- Final Plot ---
plt.ioff()
pv, sp, cv= simulate_system(kp_opt, ki_opt, kd_opt)

settling_pv = pv[:len(t_eval)]
settling_time = compute_settling_time(settling_pv, t_eval, sp[-1])
if settling_time >= sim_time:
    settling_time = 0.00

overshoot = pv - sp
overshoot[overshoot < 0] = 0
max_overshoot = np.max(overshoot)
max_overshoot_pct = (max_overshoot/(pv_max-pv_min))*100
print(f"📈 Maximum overshoot: {max_overshoot:.3f}/{max_overshoot_pct:.3f}%")
print(f"📏 Settling Time: {settling_time:.2f} seconds")

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
df.to_csv('combined_output.csv', index=False)

endtime = time.time()
print(f"Elapsed Time: {(endtime - starttime) / 60:.2f} minutes")
#print(cv_buffer[0:50])

plt.show()
