# ============================================================
# utils/physics.py
# v2 - Motor RK4 con soporte para fuerzas externas (Viento)
# ============================================================

import numpy as np
from numba import njit
from utils.quaternion_utils import (
    quat_multiply,
    quat_normalize,
    quat_to_rotation_matrix,
    quat_derivative
)

@njit(cache=True)
def actions_to_thrusts(actions, f_min, f_max):
    thrusts = np.empty(4)
    for i in range(4):
        a = actions[i]
        if a < -1.0: a = -1.0
        elif a > 1.0: a = 1.0
        thrusts[i] = f_min + (a + 1.0) / 2.0 * (f_max - f_min)
    return thrusts

@njit(cache=True)
def compute_forces_and_torques(thrusts, arm_length, torque_coeff):
    F_total = thrusts[0] + thrusts[1] + thrusts[2] + thrusts[3]
    force_body = np.array([0.0, 0.0, -F_total])
    tau_x = arm_length * (thrusts[3] - thrusts[1])
    tau_y = arm_length * (thrusts[0] - thrusts[2])
    tau_z = torque_coeff * (thrusts[0] - thrusts[1] + thrusts[2] - thrusts[3])
    torques = np.array([tau_x, tau_y, tau_z])
    return force_body, torques

@njit(cache=True)
def state_derivative(state_13, force_body, torques, mass, gravity, inertia, wind_force):
    quat = state_13[3:7]
    vel = state_13[7:10]
    omega = state_13[10:13]

    d_state = np.zeros(13)

    # 1. Velocidad
    d_state[0] = vel[0]
    d_state[1] = vel[1]
    d_state[2] = vel[2]

    # 2. Cuaternión
    dq = quat_derivative(quat, omega)
    d_state[3] = dq[0]
    d_state[4] = dq[1]
    d_state[5] = dq[2]
    d_state[6] = dq[3]

    # 3. Aceleración lineal con VIENTO
    R = quat_to_rotation_matrix(quat)
    f_inertial_x = R[0, 0]*force_body[0] + R[0, 1]*force_body[1] + R[0, 2]*force_body[2]
    f_inertial_y = R[1, 0]*force_body[0] + R[1, 1]*force_body[1] + R[1, 2]*force_body[2]
    f_inertial_z = R[2, 0]*force_body[0] + R[2, 1]*force_body[1] + R[2, 2]*force_body[2]

    d_state[7] = (f_inertial_x + wind_force[0]) / mass
    d_state[8] = (f_inertial_y + wind_force[1]) / mass
    d_state[9] = (f_inertial_z + mass * gravity + wind_force[2]) / mass

    # 4. Aceleración angular
    p, q, r = omega[0], omega[1], omega[2]
    Ixx, Iyy, Izz = inertia[0], inertia[1], inertia[2]
    d_state[10] = (torques[0] - (Izz - Iyy) * q * r) / Ixx
    d_state[11] = (torques[1] - (Ixx - Izz) * p * r) / Iyy
    d_state[12] = (torques[2] - (Iyy - Ixx) * p * q) / Izz

    return d_state

@njit(cache=True)
def rk4_step(state_13, force_body, torques, mass, gravity, inertia, dt, wind_force):
    k1 = state_derivative(state_13, force_body, torques, mass, gravity, inertia, wind_force)
    s2 = state_13 + 0.5 * dt * k1
    k2 = state_derivative(s2, force_body, torques, mass, gravity, inertia, wind_force)
    s3 = state_13 + 0.5 * dt * k2
    k3 = state_derivative(s3, force_body, torques, mass, gravity, inertia, wind_force)
    s4 = state_13 + dt * k3
    k4 = state_derivative(s4, force_body, torques, mass, gravity, inertia, wind_force)

    new_state = state_13 + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

    q = new_state[3:7]
    q_norm = quat_normalize(q)
    new_state[3:7] = q_norm

    return new_state

@njit(cache=True)
def simulate_step(state_13, actions, mass, gravity, arm_length,
                  torque_coeff, inertia, f_min, f_max, dt, substeps, wind_force):
    thrusts = actions_to_thrusts(actions, f_min, f_max)
    force_body, torques = compute_forces_and_torques(thrusts, arm_length, torque_coeff)

    for _ in range(substeps):
        state_13 = rk4_step(
            state_13, force_body, torques,
            mass, gravity, inertia, dt, wind_force
        )

    return state_13