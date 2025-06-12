import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.optimize import differential_evolution

# Constants
points_per_second = 50
sim_time = 20
num_points = int(sim_time * points_per_second)
t_eval = np.linspace(0, sim_time, num_points)
timestep = t_eval[1] - t_eval[0]
settling_tolerance = 0.005

# PID Controller Class
class PID:
    # Initialize PID, set parameters and set integral and error to zero
    def __init__(self, Kp, Ki, Kd, cv_start):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral = cv_start / self.Ki if self.Ki > 0 else 0.0
        self.prev_error = 0

    # Reset PID, set integral and error to zero
    def reset(self, cv_start):
        self.integral = cv_start / self.Ki if self.Ki > 0 else 0.0
        self.prev_error = 0
        
    # Core PID logic  
    def update(self, error, dt, cv_min, cv_max):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        control = np.clip(self.Kp * error + self.Ki * self.integral + self.Kd * derivative, cv_min, cv_max)
        self.prev_error = error
        integral = self.integral
        return control, integral


# System Simulation
def simulate_system(Kp, Ki, Kd, cv_min, cv_max, cv_start, pv_min, pv_max, pv_start, pv_final, process_gain, dead_time, time_constant):
    # Call the PID and make sure it's reset at the start
    pid = PID(Kp, Ki, Kd, cv_start)
    pid.reset(cv_start)

    # Initialize lists for cv/pv/sp histories
    cv_history = []
    pv_array = []
    sp_array = []
    
    # Initialize CV and PV based on setpoints
    pv = np.clip(pv_start, pv_min, pv_max)
    # Fill out the cv buffer with the CV init value to avoid starting at zero
    cv_delay_steps = max(1, int(np.round(dead_time / timestep)))
    cv_buffer = [cv_start] * cv_delay_steps
    
    disturbance_magnitude = 0.001
    disturbance_time = 5
    disturbance_duration = 3.0
    disturbance_end = disturbance_time + disturbance_duration
    disturbance_value = pv_final * disturbance_magnitude
    sp = pv_final
    # Run the actual simulation for x time
    for i, t in enumerate(t_eval):
        # Update the PID every cycle
        error = sp - pv
        cv, integral = pid.update(error, timestep, cv_min, cv_max)
        
        cv_buffer.append(cv)
        delayed_cv = cv_buffer.pop(0)
        # Update the PID simulation
        dpv = (-pv + process_gain * delayed_cv) / time_constant
        pv += dpv * timestep
        pv = np.clip(pv, pv_min, pv_max)
        
        # Add the PV, cv and SP to arrays for plotting
        pv_array.append(pv)
        cv_history.append(cv)
        sp_array.append(sp)
    return np.array(pv_array), np.array(sp_array), np.array(cv_history)

# Compute Settling Time
def compute_settling_time(pv, t, pv_final, tolerance=settling_tolerance):
    pv = np.asarray(pv)
    t = np.asarray(t)

    min_len = min(len(pv), len(t))
    pv = pv[:min_len]
    t = t[:min_len]

    band_upper = pv_final * (1 + tolerance)
    band_lower = pv_final * (1 - tolerance)

    for i in range(min_len):
        window = pv[i:]
        if np.all((window >= band_lower) & (window <= band_upper)):
            return t[i]
    return t[-1]  # If it never settles

trial_counter = 0

# GUI application
class PIDApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PID Simulation with Optimization")
        self.geometry("1200x600")
        self.kp_var = tk.DoubleVar(value=0.185)
        self.ki_var = tk.DoubleVar(value=0.7)
        self.kd_var = tk.DoubleVar(value=0.0)
        self.cv_min = tk.DoubleVar(value=0.0)
        self.cv_max = tk.DoubleVar(value=100.0)
        self.cv_start = tk.DoubleVar(value=30.0)
        self.cv_final = tk.DoubleVar(value=70.0)
        self.pv_min = tk.DoubleVar(value=0.0)
        self.pv_max = tk.DoubleVar(value=250.0)
        self.pv_start = tk.DoubleVar(value=10.0)
        self.pv_final = tk.DoubleVar(value=45.0)
        self.dead_time = tk.DoubleVar(value=0.1)
        self.tau = tk.Doublevar(value=0.75)
        self.time_constant = self.compute_time_constant()
        self.process_gain = self.compute_process_gain()
        self.disturbance = tk.BooleanVar(value=True)
        self.noise = tk.BooleanVar(value=False)
        self.create_widgets()
        self.init_plot()

    def create_widgets(self):
        process = ttk.LabelFrame(self, text="Process")
        process.grid(row=0, column=0, sticky="ns", padx=5, pady=5)
        
        ttk.Label(process, text="Control Variable").grid(row=0, column=0, stick="e")
        
        ttk.Label(process, text="CV Min").grid(row=1, column=0, stick="e")
        ttk.Entry(process, textvariable=self.cv_min).grid(row=1, column=1)
        
        ttk.Label(process, text="CV Max").grid(row=2, column=0, stick="e")
        ttk.Entry(process, textvariable=self.cv_max).grid(row=2, column=1)
        
        ttk.Label(process, text="CV Start").grid(row=3, column=0, stick="e")
        ttk.Entry(process, textvariable=self.cv_start).grid(row=3, column=1)
        
        ttk.Label(process, text="CV Final").grid(row=4, column=0, stick="e")
        ttk.Entry(process, textvariable=self.cv_final).grid(row=4, column=1)
        
        ttk.Label(process, text="Process Variable").grid(row=5, column=0, stick="e")
        
        ttk.Label(process, text="PV Min").grid(row=6, column=0, stick="e")
        ttk.Entry(process, textvariable=self.pv_min).grid(row=6, column=1)
        
        ttk.Label(process, text="PV Max").grid(row=7, column=0, stick="e")
        ttk.Entry(process, textvariable=self.pv_max).grid(row=7, column=1)
        
        ttk.Label(process, text="PV Start").grid(row=8, column=0, stick="e")
        ttk.Entry(process, textvariable=self.pv_start).grid(row=8, column=1)
        
        ttk.Label(process, text="PV Final").grid(row=9, column=0, stick="e")
        ttk.Entry(process, textvariable=self.pv_final).grid(row=9, column=1)
        
        ttk.Label(process, text="Process Simulation").grid(row=10, column=0, stick="e")
        
        ttk.Label
        
        controls = ttk.LabelFrame(self, text="Controls")
        controls.grid(row=0, column=1, sticky="ns", padx=5, pady=5)

        ttk.Label(controls, text="Kp:").grid(row=0, column=0, sticky="e")
        ttk.Entry(controls, textvariable=self.kp_var).grid(row=0, column=1)

        ttk.Label(controls, text="Ki:").grid(row=1, column=0, sticky="e")
        ttk.Entry(controls, textvariable=self.ki_var).grid(row=1, column=1)

        ttk.Label(controls, text="Kd:").grid(row=2, column=0, sticky="e")
        ttk.Entry(controls, textvariable=self.kd_var).grid(row=2, column=1)

        ttk.Checkbutton(controls, text="Disturbance", variable=self.disturbance).grid(row=3, column=0, sticky="w")
        ttk.Checkbutton(controls, text="Measurement Noise", variable=self.noise).grid(row=4, column=0, sticky="w")

        ttk.Button(controls, text="Run Simulation", command=self.run_manual).grid(row=5, column=0, columnspan=2, pady=5)
        ttk.Button(controls, text="Optimize PID", command=self.run_optimization).grid(row=6, column=0, columnspan=2, pady=5)

        self.plot_frame = ttk.LabelFrame(self, text="Live Plot")
        self.plot_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)

    def init_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.ax.set_title("Live PID Optimization")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Value")
        self.pv_line, = self.ax.plot([], [], label="PV")
        self.sp_line, = self.ax.plot([], [], 'r--', label="SP")
        self.cv_line, = self.ax.plot([], [], 'g:', label="CV")
        self.ax.legend()
        self.fig.tight_layout()
        
    def update_plot(self, t, pv, sp, cv, title):
        self.ax.set_title(title)
        self.pv_line.set_data(t, pv)
        self.sp_line.set_data(t, sp)
        self.cv_line.set_data(t, cv)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
        self.canvas.flush_events()
        
    def compute_process_gain(self):
        return (self.pv_final.get() - self.pv_start.get()) / (self.cv_final.get() - self.cv_start.get())
    
    def compute_time_constant(self):
        return (max(1.1 * (self.tau.get() - self.dead_time.get()), 0.1))

    def run_manual(self):
        kp = self.kp_var.get()
        ki = self.ki_var.get()
        kd = self.kd_var.get()
        pv, sp, cv = simulate_system(kp, ki, kd, self.cv_min.get(), self.cv_max.get(), self.cv_start.get(), self.pv_min.get(), self.pv_max.get(), self.pv_start.get(), self.pv_final.get(), self.process_gain, self.dead_time.get(), self.time_constant.get())
        title = "PID Simulation"
        self.update_plot(t_eval, pv, sp, cv, title)

    def run_optimization(self):
        # Create live plot window
        def live_cost_function(pid_gains):
            global trial_counter
            Kp, Ki, Kd = pid_gains
            
            # Run the system Simulation
            pv, sp, cv= simulate_system(Kp, Ki, Kd, self.cv_min.get(), self.cv_max.get(), self.cv_start.get(), self.pv_min.get(), self.pv_max.get(), self.pv_start.get(), self.pv_final.get(), self.process_gain, self.dead_time.get(), self.time_constant.get())

            # If PV is not a number then reject this solution
            if np.any(np.isnan(pv)) or np.any(np.isinf(pv)):
                return 1e6
            
            # Live plotting
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
            title = (f"Trial #{trial_counter+1} | Kp={Kp:.3f}, Ki={Ki:.3f}, Kd={Kd:.3f}")
            cost = (alpha * ise) + (beta * overshoot_penalty) + (gamma * initial_instability_penalty) + (delta * initial_cv_penalty) + (epsilon * settling_time) 
            self.update_plot(t_eval, pv, sp, cv, title)
            return cost

        bounds = [(0, 2), (0, 2), (0, 1)]
        result = differential_evolution(live_cost_function, bounds, seed=42, strategy='best1bin', maxiter=100, popsize=15, tol=1e-6)

        # Final result
        kp, ki, kd = result.x
        pv, sp, cv = simulate_system(kp, ki, kd, self.cv_min.get(), self.cv_max.get(), self.cv_start.get(), self.pv_min.get(), self.pv_max.get(), self.pv_start.get(), self.pv_final.get(), self.process_gain, self.dead_time.get(), self.time_constant.get())
        title = (f"Final PID Optimization | Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}")
        self.update_plot(t_eval, pv, sp, cv, title)

# Launch GUI
if __name__ == "__main__":
    app = PIDApp()
    app.mainloop()
