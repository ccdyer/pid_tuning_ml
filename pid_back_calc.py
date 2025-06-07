import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('combined_output.csv')
t_array = df['Timestamp'].to_numpy()
pv_array = df['PV'].to_numpy()
cv_array = df['CV'].to_numpy()
sp_array = df['SP'].to_numpy()

# Calculate error
df["error"] = df["SP"] - df["PV"]

# Sampling time
dt = df["Timestamp"].diff().fillna(0)

# Integral of error (cumulative sum approximation)
df["error_int"] = (df["error"] * dt).cumsum()

# Derivative of error
df["error_deriv"] = df["error"].diff() / dt
df["error_deriv"] = df["error_deriv"].fillna(0)

X = df[["error", "error_int", "error_deriv"]]
y = df["CV"]

model = LinearRegression()
model.fit(X, y)

Kp, Ki, Kd = model.coef_
bias = model.intercept_

print(f"Estimated PID Gains:\nKp = {Kp:.3f}, Ki = {Ki:.3f}, Kd = {Kd:.3f}, Bias = {bias:.3f}")

df["CV_pred"] = model.predict(X)

plt.plot(df["Timestamp"], df["CV"], label="Actual CV")
plt.plot(df["Timestamp"], df["CV_pred"], label="Predicted CV (PID)", linestyle="--")
plt.legend()
plt.xlabel("Timestamp")
plt.ylabel("Control Variable (CV)")
plt.title("Controller Model Fit")
plt.grid(True)
plt.show()