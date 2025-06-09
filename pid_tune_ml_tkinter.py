import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.optimize import differential_evolution

# Constants
cv_min, cv_max = 0, 100
pv_min, pv_max = 0, 250
setpoint = 40
dead_time = 0.1
tangent_time = 0.75
points_per_second = 50
sim_time = 20
num_points = int(sim_time * points_per_second)
t_eval = np.linspace(0, sim_time, num_points)
timestep = t_eval[1] - t_eval[0]
settling_tolerance = 0.005
cv_init = 47
pv_init = 40
process_gain = (45.0 - 10.0) / (70.0 - 30.0)
time_constant = max(1.1 * (tangent_time - dead_time), 0.1)

# PID controller class
class PID:
    def __init__(self, Kp, Ki, Kd, cv_init):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral = cv_init / Ki if Ki > 0 else 0.0
        self.prev_error = 0

    def reset(self):
        self.integral = cv_init / self.Ki if self.Ki > 0 else 0.0
        self.prev_error = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        control = np.clip(self.Kp * error + self.Ki * self.integral + self.Kd * derivative, cv_min, cv_max)
        self.prev_error = error
        return control

# System simulation
def simulate_system(Kp, Ki, Kd, disturbance, noise):
    pid = PID(Kp, Ki, Kd, cv_init)
    pid.reset()
    cv_buffer = [cv_init] * max(1, int(dead_time / timestep))
    pv = np.clip(pv_init, pv_min, pv_max)

    pv_array, sp_array, cv_array = [], [], []
    for i, t in enumerate(t_eval):
        sp = setpoint - (setpoint * 0.001) if disturbance and 5 <= t < 8 else setpoint
        error = sp - pv
        cv = pid.update(error, timestep)
        cv_buffer.append(cv)
        delayed_cv = cv_buffer.pop(0)
        dpv = (-pv + process_gain * delayed_cv) / time_constant
        pv += dpv * timestep
        pv = np.clip(pv, pv_min, pv_max)
        if noise:
            pv += np.random.normal(0, 0.01)
        pv_array.append(pv)
        cv_array.append(cv)
        sp_array.append(sp)
    return np.array(t_eval), np.array(pv_array), np.array(sp_array), np.array(cv_array)

def compute_settling_time(pv, t, sp, tol=settling_tolerance):
    band_upper = sp * (1 + tol)
    band_lower = sp * (1 - tol)
    for i in range(len(pv)):
        if np.all((pv[i:] >= band_lower) & (pv[i:] <= band_upper)):
            return t[i]
    return t[-1]

def cost_function(params, disturbance, noise):
    Kp, Ki, Kd = params
    if Kp < 0 or Ki < 0 or Kd < 0:
        return 1e6
    t, pv, sp, _ = simulate_system(Kp, Ki, Kd, disturbance, noise)
    overshoot = max(0, np.max(pv - sp))
    settling = compute_settling_time(pv, t, sp[-1])
    return 5 * overshoot + settling  # weighted cost

# GUI application
class PIDApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PID Simulation with Optimization")
        self.geometry("500x400")
        self.kp_var = tk.DoubleVar(value=0.185)
        self.ki_var = tk.DoubleVar(value=0.7)
        self.kd_var = tk.DoubleVar(value=0.0)
        self.disturbance = tk.BooleanVar(value=True)
        self.noise = tk.BooleanVar(value=False)
        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self, text="Kp:").grid(row=0, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.kp_var).grid(row=0, column=1)

        ttk.Label(self, text="Ki:").grid(row=1, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.ki_var).grid(row=1, column=1)

        ttk.Label(self, text="Kd:").grid(row=2, column=0, sticky="e")
        ttk.Entry(self, textvariable=self.kd_var).grid(row=2, column=1)

        ttk.Checkbutton(self, text="Enable Disturbance", variable=self.disturbance).grid(row=3, column=0, columnspan=2)
        ttk.Checkbutton(self, text="Enable Noise", variable=self.noise).grid(row=4, column=0, columnspan=2)

        ttk.Button(self, text="Run Manual Simulation", command=self.run_manual).grid(row=5, column=0, columnspan=2, pady=10)
        ttk.Button(self, text="Optimize PID", command=self.run_optimization).grid(row=6, column=0, columnspan=2, pady=5)

    def run_manual(self):
        kp = self.kp_var.get()
        ki = self.ki_var.get()
        kd = self.kd_var.get()
        t, pv, sp, cv = simulate_system(kp, ki, kd, self.disturbance.get(), self.noise.get())
        self.show_plot(t, pv, sp, cv, kp, ki, kd)

    def run_optimization(self):
        # Create live plot window
        self.live_fig, self.live_ax = plt.subplots(figsize=(7, 3))
        self.live_plot_win = tk.Toplevel(self)
        self.live_plot_win.title("Optimization Progress")
        self.live_canvas = FigureCanvasTkAgg(self.live_fig, master=self.live_plot_win)
        self.live_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.live_ax.set_title("Live PID Optimization")
        self.live_ax.set_xlabel("Time (s)")
        self.live_ax.set_ylabel("PV / SP / CV")
        self.live_ax.grid(True)
        self.live_lines = {
            'pv': self.live_ax.plot([], [], label="PV")[0],
            'sp': self.live_ax.plot([], [], 'r--', label="SP")[0],
            'cv': self.live_ax.plot([], [], 'g:', label="CV")[0],
        }
        self.live_ax.legend()
        self.live_fig.tight_layout()
        plt.ion()
        plt.show()

        def live_cost_function(params):
            Kp, Ki, Kd = params
            if Kp < 0 or Ki < 0 or Kd < 0:
                return 1e6
            t, pv, sp, cv = simulate_system(Kp, Ki, Kd, self.disturbance.get(), self.noise.get())

            # Update live plot
            self.live_lines['pv'].set_data(t, pv)
            self.live_lines['sp'].set_data(t, sp)
            self.live_lines['cv'].set_data(t, cv)
            self.live_ax.relim()
            self.live_ax.autoscale_view()
            self.live_canvas.draw()
            self.live_canvas.flush_events()
            plt.pause(0.001)

            overshoot = max(0, np.max(pv - sp))
            settling = compute_settling_time(pv, t, sp[-1])
            return 5 * overshoot + settling

        bounds = [(0, 2), (0, 2), (0, 1)]
        result = differential_evolution(live_cost_function, bounds)

        # Final result
        kp, ki, kd = result.x
        self.kp_var.set(kp)
        self.ki_var.set(ki)
        self.kd_var.set(kd)
        t, pv, sp, cv = simulate_system(kp, ki, kd, self.disturbance.get(), self.noise.get())
        self.show_plot(t, pv, sp, cv, kp, ki, kd)

        plt.ioff()
        self.live_plot_win.destroy()

    def show_plot(self, t, pv, sp, cv, kp, ki, kd):
        settling = compute_settling_time(pv, t, sp[-1])
        overshoot = max(0, np.max(pv - sp))
        overshoot_pct = (overshoot / (pv_max - pv_min)) * 100

        plot_win = tk.Toplevel(self)
        plot_win.title("Simulation Results")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, pv, label="PV")
        ax.plot(t, sp, 'r--', label="Setpoint")
        ax.plot(t, cv, 'g:', label="CV")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Values")
        ax.set_title(f"Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f} | Settling: {settling:.2f}s | Overshoot: {overshoot_pct:.2f}%")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=plot_win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Launch GUI
if __name__ == "__main__":
    app = PIDApp()
    app.mainloop()
