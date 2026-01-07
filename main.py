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


""" 
------------------------
       Input Data 
------------------------
"""


# Material & Geometry
E   = 210e9     # Modulus
rho = 7850
L   = 2.0
R   = 0.02

A = np.pi * R**2        # Surface 
I = np.pi * R**4/4.0    # Bending moment

# Input Data
P = 1e3                 # Force
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
