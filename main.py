# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 11:13:13 2026
@author: Micha

Computational Mechanics II - 2025/26
3rd Homework: Dynamic Beam Analysis
by: Michaela Alexandridi
"""
import numpy as np

"""

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
Part 1: Static Solver
"""
