# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 11:13:13 2026
@author: Micha

Computational Mechanics II - 2025/26
3rd Homework: Dynamic Beam Analysis
by: Michaela Alexandridi
"""

import numpy as np
from assembler import assemble_global_matrices, apply_bc
from static_solver import tip_load_vector, static_solver, analytical_solution
from newmark import newmark, frequency_estimation
import matplotlib.pyplot as plt
from postprocessing import animate_beam


""" 
------------------------
       Input Data 
------------------------
"""


# Material & Geometry
E   = 210e9     # Modulus
rho = 7850      # Density
L   = 2.0       # Length
R   = 0.02      # Beam Radius

A = np.pi * R**2        # Surface 
I = np.pi * R**4/4.0    # Bending moment

# Input Data
P = -1e3                # Force
n_elem = 8              # Number of elements



"""
--------------------------------
    Part 1: Static Analysis
--------------------------------
"""

# Assembling System Ku + Mu = f
K, M = assemble_global_matrices(n_elem, E, I, rho, A, L)
f = tip_load_vector(n_elem, P)

# Define Boundary Conditions
fixed_dofs = [0, 1]
K_r, M_r, f_r, free = apply_bc(K, M, f, fixed_dofs)

# Solve System
u = static_solver(K_r, f_r)
wL_numerical = u[-2]
wL_analytical, _ = analytical_solution(P, L, E, I)

# Check results
print(f"Static Tip Displacement FEM: {wL_numerical:.6e}")
print(f"Static Tip Displacement FEM: {wL_analytical:.6e}")



"""
--------------------------------
    Part 2: Free Vibration
--------------------------------
"""

# Input Newmark data
u0 = u
v0 = np.zeros_like(u0)

zero_force = lambda t: np.zeros_like(u0)

dt = 1e-5
t_end = 2.0

# Newmark Integration
t, uh, vh, ah = newmark(M_r, K_r, zero_force, u0, v0, dt, t_end)

# Plot results
plt.figure(figsize=(9,4))
plt.plot(t,uh[:,-2])
plt.xlabel("t")
plt.ylabel("w(L)")
plt.show()

# Omega
omega_numerical = frequency_estimation(t, uh[:,-2], min_height=0.0, min_distance=0.08)
omega_analytical = (1.875**2) * np.sqrt(E*I/(rho*A*L**4))

print(f"ω1 FEM: {omega_numerical:.3f}")
print(f"ω1 ANA: {omega_analytical:.3f}")



"""
--------------------------------
    Part 3: Dynamic Analysis
--------------------------------
"""

# Input
P0 = 1e3
omega = 0.95*omega_numerical                                      # Given frequency
force = lambda t: (np.eye(len(u0))[-2] * P0 * np.sin(omega*t))    # Time-varying load

"""
B way for calculating harmonic force
    
def force(t):
    f = np.zeros_like(u0)
    f[-2] = P0*np.sin(omega*t)
    
    return f

"""

# Initialise
new_u0 = np.zeros_like(u0)
new_v0 = np.zeros_like(new_u0)

# Newmark Integration
new_t, new_uh, new_vh, new_ah = newmark(M_r, K_r, force, new_u0, new_v0, dt, t_end)

# Create GIF file for u
animate_beam(new_uh,
             free_dofs=free,
             n_elem=n_elem,
             L=L,
             time=t,
             scale=55,
             filename="semfebeam.gif",
             fps0=30,
             max_frames=150,
             show=True)