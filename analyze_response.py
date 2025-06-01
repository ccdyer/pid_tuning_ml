import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time
import pandas as pd

df = pd.read_csv('combined_output.csv')
t_array = df['Timestamp'].to_numpy()
pv_array = df['PV'].to_numpy()
cv_array = df['CV'].to_numpy()

array = df.to_numpy()
t_prev = 0
pv_prev = 0
cv_prev = 0
min_pct_change = 0.1
step_change_time = -1
dead_time_final = -1
tolerance_pct = 0.005
window_size = 20

for i, x in enumerate(array):
    t = x[0]
    pv = x[1]
    cv = x[2]
    if t != 0 and step_change_time == -1:
        if cv >= (cv_prev * (1+min_pct_change)):
            step_change_time = t
            cv_initial = cv_prev
            cv_final = cv
            pv_initial = pv
            pv_response_threshold = pv_initial * (tolerance_pct+1)
            step_change_index = i
    if step_change_time != -1:
        if pv >= pv_response_threshold and dead_time_final == -1:
            dead_time_final = t
            dead_time = dead_time_final - step_change_time
    
    t_prev = t
    pv_prev = pv
    cv_prev = cv
step_change_array = pv_array[step_change_index:]
for i in range(len(step_change_array) - window_size):
    window = step_change_array[i:i+window_size]
    if np.all(np.abs(window - window[-1]) < tolerance_pct * abs(window[-1])):
        pv_final = np.mean(window)
        #print(f"Final PV (settled at index {i}): {pv_final:.3f}")
        break
else:
    print(" PV Never Stabilized")
    
process_gain = (pv_final-pv_initial) / (cv_final - cv_initial)
pv_63 = pv_initial + (0.632 * (pv_final - pv_initial))
pv_63_time = -1
for i, x in enumerate(array):
    t = x[0]
    pv = x[1]
    cv = x[2]
    if pv >= pv_63 and pv_63_time == -1:
        pv_63_time = t
time_constant = pv_63_time - step_change_time

print(process_gain)
print(dead_time)
print(time_constant)
