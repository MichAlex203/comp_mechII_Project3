# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 13:17:16 2026
@author: Micha

Newmark Integration
"""
import numpy as np
import matplotlib.pyplot as plt

# --------------
# Newmark Method
# --------------

def newmark(M, K, f_func, u0, v0, dt, t_end, alpha=0.25, beta=0.5):
    n = len(u0)
    n_steps = int(t_end/dt)
    t = np.linspace(0, t_end, n_steps)
    
    # Initialise
    u = np.zeros((n_steps,n))   # Displacement
    v = np.zeros((n_steps,n))   # Velocity
    a = np.zeros((n_steps,n))   # Acceleration
    
    # Initiative values
    u[0] = u0
    v[0] = v0
    a[0] = np.linalg.solve(M, f_func(0)-K@u0)
    
    # Apply Newmark Method
    K_eff = K + M/(alpha*dt**2)
    K_eff_inv = np.linalg.inv(K_eff)
    
    for i in range(1,n_steps):
        f_eff = (
            f_func(t[i]) + 
            M @ (
                u[i-1]/(alpha*dt**2) + 
                v[i-1]/(alpha*dt) + 
                (1/(2*alpha)-1)*a[i-1]
                )
            )
        
        # Update u, v, a
        u[i] = K_eff_inv @ f_eff
        a[i] = (
            (u[i] - u[i-1])/(alpha*dt**2)
            - v[i-1]/(alpha*dt)
            - (1/(2*alpha)-1)*a[i-1]
            )
        v[i] = v[i-1] + dt*((1-beta)*a[i-1]+beta*a[i])
        
    return t, u, v, a


# ---------------------------
# Estimation of natural frequency
# ---------------------------

def frequency_estimation(t, uhl, min_height=1e-8, min_distance=0.01, n_periods=5):
    
    # Find peaks  
    peaks_all = np.where(
        (uhl[1:-1] > uhl[:-2]) &
        (uhl[1:-1] > uhl[2:]) &
        (uhl[1:-1] > min_height)
        )[0] + 1
    
    if len(peaks_all) < 2:
        raise ValueError("Not enough positive peaks")
        
    # Filter close peaks
    peak_times = []
    peak_values = []
    last_peak_time = -np.inf
    
    for idx in peaks_all:
        t_peak = t[idx]
        if t_peak - last_peak_time >= min_distance:
            peak_times.append(t_peak)
            peak_values.append(uhl[idx])
            last_peak_time = t_peak
    
    # Calculate natural frequency
    peak_times = np.array(peak_times)
    peak_values = np.array(peak_values)
    
    periods = np.diff(peak_times)
    T_avg = np.mean(periods)
    
    omega_est = 2*np.pi/T_avg
    
    # Plot
    plt.figure(figsize=(9,4))
    plt.plot(t,uhl)
    plt.plot(peak_times,peak_values,"ro")
    plt.xlabel("t")
    plt.ylabel("w(L)")
    plt.show()
    
    return omega_est
    