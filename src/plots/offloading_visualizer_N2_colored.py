#!/usr/bin/env python3
"""
Constrained Nonlinear Aggregative Offloading Problem (N=2)
3D visualization with BOTH feasible and infeasible regions shown
in different colors.

Feasible:  sigma(z1,z2) <= c
Infeasible: sigma(z1,z2) > c
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Parameters (edit freely)
# -------------------------
w1 = 1.0
w2 = 1.0
rho = 0.5
c = 0.8

E1_tx = 2
E2_tx = 2
T1_tx = 2
T2_tx = 2

E1_task = 4
E2_task = 4
T1_task = 3
T2_task = 3

def l_loc_1():
    return E1_task + T1_task

def l_loc_2():
    return E2_task + T2_task


kappa = 1

gamma1 = 0.5
gamma2 = 0.5

# -------------------------
# Model definitions
# -------------------------
def phi1(z1):
    return w1 * (z1 + rho * z1**2)

def phi2(z2):
    return w2 * (z2 + rho * z2**2)
    # return w2 * (np.exp(z2)-1)

def sigma(z1, z2):
    return (phi1(z1) + phi2(z2)) / 2.0

def tau1(z1, s):
    # return kappa * 1 / (1 + np.exp(-s))
    # return s / (z1 + 1e-10)
    return kappa * s

def tau2(z2, s):
    # return kappa * 1 / (1 + np.exp(-s))
    # return s / (z2 + 1e-10)
    return kappa * s

def l_cloud_1(z1, s):
    return E1_tx + T1_tx

def l_cloud_2(z2, s):
    return E2_tx + T2_tx

def ell1(z1, s):
    return (1.0 - z1) * l_loc_1() + z1 * l_cloud_1(z1, s) + 0.5*tau1(z1, s) + 0.5 * gamma1 * z1**2

def ell2(z2, s):
    return (1.0 - z2) * l_loc_2() + z2 * l_cloud_2(z2, s) + 0.5*tau2(z2, s) + 0.5 * gamma2 * z2**2

def F(z1, z2):
    s = sigma(z1, z2)
    return ell1(z1, s) + ell2(z2, s)

# -------------------------
# Visualization
# -------------------------
def plot_3d(n_grid=151):

    z1 = np.linspace(-5.0, 5.0, n_grid)
    z2 = np.linspace(-5.0, 5.0, n_grid)
    Z1, Z2 = np.meshgrid(z1, z2)

    S = sigma(Z1, Z2)
    Cost = F(Z1, Z2)

    feasible = (S <= c)
    infeasible = (S > c)

    Cost_feasible = np.where(feasible, Cost, np.nan)
    Cost_infeasible = np.where(infeasible, Cost, np.nan)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot feasible region in blue
    ax.plot_surface(Z1, Z2, Cost_feasible)

    # Plot infeasible region in red
    ax.plot_surface(Z1, Z2, Cost_infeasible)

    # Draw boundary sigma = c projected on floor
    z_floor = np.nanmin(Cost) - 0.05 * (np.nanmax(Cost) - np.nanmin(Cost))
    ax.contour(Z1, Z2, S, levels=[c], zdir='z', offset=z_floor)

    ax.set_xlabel("z1 (agent 1)")
    ax.set_ylabel("z2 (agent 2)")
    ax.set_zlabel("Total cost F(z1,z2)")

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_3d()
